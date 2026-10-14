"""iCuerious Product-to-Patent Mapping: secure client portal.  Run with:  streamlit run app.py"""
import hashlib
import html
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import engine as E
import security as S
import ui

st.set_page_config(page_title="iCuerious | Product-to-Patent Mapping", layout="wide", initial_sidebar_state="collapsed")
ui.inject_theme()

APP_DIR = Path(__file__).parent
CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", APP_DIR / "corpus"))       # chunks.json, embeddings.npy, manifest.json
EMB_DIR = Path(os.environ.get("EMB_CACHE_DIR", APP_DIR / "emb_cache"))    # precomputed patent-limitation embeddings
PERSIST_REVIEWS = str(S.get_secret("PERSIST_REVIEWS", "true")).lower() != "false"


class UserError(Exception):
    """A message that is safe to show to the client as-is."""


# ============================================================================ sign-in (fails closed)
user = S.login_gate()
is_admin = user["role"] == "admin"
if not is_admin:
    st.markdown(ui.HIDE_SIDEBAR, unsafe_allow_html=True)
if ui.header(user["name"], user["role"]):
    S.logout()

limits = S.get_limits()
api_key = S.get_secret("GEMINI_API_KEY")                      # server-side only; never rendered, never typed in the UI
cfg = dict(E.DEFAULTS)
cfg["gemini_model"] = S.get_secret("GEMINI_MODEL", cfg["gemini_model"])
try:                                                           # the query model must match the one that embedded the corpus
    cfg["embed_model"] = json.loads((CORPUS_DIR / "manifest.json").read_text()).get("embed_model") or cfg["embed_model"]
except Exception:
    pass


# ============================================================================ cached server-side resources
@st.cache_resource(show_spinner=False)
def corpus_from_disk(chunks_path: str, emb_path: str, manifest_path: str, mtime: float):
    return E.load_corpus(chunks_path, emb_path, manifest_path if Path(manifest_path).exists() else None)


@st.cache_resource(show_spinner=False)
def corpus_from_bytes(chunks_bytes: bytes, emb_bytes: bytes, manifest_bytes, digest: str):
    return E.load_corpus(chunks_bytes, emb_bytes, manifest_bytes)


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str, _corpus, n_chunks: int):
    emb = E.Embedder(model_name)
    probe = np.random.default_rng(0).choice(n_chunks, min(5, n_chunks), replace=False)
    pe = emb.encode([_corpus.chunks[i]["text"] for i in probe])
    agree = float(np.mean(np.sum(pe * _corpus.emb[probe], axis=1)))
    if pe.shape[1] != _corpus.dim or agree < 0.97:
        raise RuntimeError(f"query model does not reproduce stored vectors (cosine {agree:.3f}, dim {pe.shape[1]} vs {_corpus.dim})")
    return emb


# ============================================================================ admin-only controls
if is_admin:
    with st.sidebar:
        st.markdown("### Administration")
        cfg["gemini_model"] = st.text_input("Review model", cfg["gemini_model"])
        cfg["embed_model"] = st.text_input("Query embedding model", cfg["embed_model"])
        cfg["per_product_candidates"] = st.number_input("Candidates per product", 1, 1000, cfg["per_product_candidates"])
        cfg["llm_pair_cap"] = st.number_input("Max pairs sent for detailed review", 1, 5000, cfg["llm_pair_cap"])
        cfg["claims_per_pair"] = st.number_input("Independent claims per pair", 1, 10, cfg["claims_per_pair"])
        cfg["max_gemini_calls"] = st.number_input("Hard cap on new review calls", 1, 20000, cfg["max_gemini_calls"])
        cfg["recall_audit_pairs_per_product"] = st.number_input("Recall-audit pairs per product", 0, 20, cfg["recall_audit_pairs_per_product"])
        cfg["coverage_similarity_threshold"] = st.slider("Coverage cutoff", 0.30, 0.90, cfg["coverage_similarity_threshold"], 0.01)
        cfg["passages_per_limitation"] = st.number_input("Passages per limitation", 1, 10, cfg["passages_per_limitation"])
        cfg["limitations_per_call"] = st.number_input("Limitations per call", 1, 60, cfg["limitations_per_call"])
        cfg["pair_claim_pick"] = st.selectbox("Claim representing a pair", ["broadest", "best"])
        cfg["dedupe_families"] = st.checkbox("One patent per family", cfg["dedupe_families"])
        cfg["gemini_workers"] = st.number_input("Parallel workers", 1, 32, cfg["gemini_workers"])
        st.caption(f"API key configured: {'yes' if api_key else 'NO'}")
        if S.AUDIT_FILE.exists():
            st.download_button("Download audit log", S.AUDIT_FILE.read_bytes(), "audit.jsonl", "application/json")

# ============================================================================ product corpus (server-side)
corpus = None
chunks_p, emb_p, man_p = CORPUS_DIR / "chunks.json", CORPUS_DIR / "embeddings.npy", CORPUS_DIR / "manifest.json"
try:
    if chunks_p.exists() and emb_p.exists():
        corpus = corpus_from_disk(str(chunks_p), str(emb_p), str(man_p), emb_p.stat().st_mtime)
    elif is_admin:
        st.info("No corpus found on the server. Upload one for this session.")
        c1, c2, c3 = st.columns(3)
        up_c, up_e, up_m = (c1.file_uploader("chunks.json", type="json"), c2.file_uploader("embeddings.npy", type="npy"),
                            c3.file_uploader("manifest.json", type="json"))
        if up_c and up_e:
            cb, eb = up_c.getvalue(), up_e.getvalue()
            corpus = corpus_from_bytes(cb, eb, up_m.getvalue() if up_m else None, hashlib.md5(cb[:100000] + eb[:100000]).hexdigest())
except Exception as e:
    st.error(S.safe_error("The product library could not be loaded.", e))
if corpus is None:
    if not is_admin:
        st.error("The product library is currently unavailable. Please contact support.")
    ui.footer()
    st.stop()
# ============================================================================ portfolio input
pf_memo = st.session_state.get("pf_memo")
results_key = st.session_state.get("results_key")
res = st.session_state.get("results")
step = 3 if (res and pf_memo and results_key and results_key[0] == pf_memo[0]) else (2 if pf_memo else 1)
ui.stepper(step)


def build_portfolio(df: pd.DataFrame, key: str):
    memo = st.session_state.get("pf_memo")
    if memo and memo[0] == key:
        return memo[1]
    try:
        cols = E.resolve_columns(df, cfg)
    except KeyError:
        raise UserError("The file must contain a Patent ID column and an Independent Claims column.")
    df_use, excluded = E.filter_portfolio(df, cols, cfg)
    if len(df_use) == 0:
        raise UserError("No patents remain after status filtering.")
    if not is_admin and len(df_use) > limits["max_patents"]:
        raise UserError(f"A single analysis is limited to {limits['max_patents']:,} patents. Please split the portfolio.")
    df_use = df_use.copy()
    cc = cols["independent_claims"]
    df_use[cc] = df_use[cc].astype(str).str.slice(0, limits["max_claim_chars"])
    pf = E.Portfolio(df_use, cols)
    if not pf.limitation_records:
        raise UserError("No claims could be parsed. Check that independent claims are numbered (1. A system ...).")
    if not is_admin and len(pf.limitation_records) > limits["max_limitations"]:
        raise UserError("This portfolio is larger than a single analysis allows. Please split it.")
    hits = S.injection_hits(df_use[cc].tolist())
    st.session_state["pf_memo"] = (key, pf, len(excluded), hits)
    st.session_state.pop("results", None)
    S.audit("portfolio_loaded", patents=len(pf.patents), limitations=len(pf.limitation_records), injection_flags=hits)
    return pf


with st.container(border=True):
    ui.section("1. Upload your patent portfolio",
               "Excel or CSV with a Patent ID column and an Independent Claims column. Files are processed in memory and not retained.")
    tab_up, tab_paste = st.tabs(["Upload file", "Paste claims"])
    df_in, df_key = None, None
    with tab_up:
        up = st.file_uploader("Portfolio file (.xlsx or .csv)", type=["xlsx", "csv"], label_visibility="collapsed")
        if up:
            data = up.getvalue()
            err = S.validate_upload(up.name, data, limits)
            if err:
                st.error(err)
            else:
                try:
                    df_in, df_key = E.read_portfolio_file(up.name, data), hashlib.sha256(data).hexdigest()
                except Exception as e:
                    st.error(S.safe_error("The file could not be read.", e))
    with tab_paste:
        pid_in = st.text_input("Patent number")
        title_in = st.text_input("Title (optional)")
        claims_in = st.text_area("Independent claim(s), numbered, e.g. '1. A system comprising: ...'", height=200)
        if claims_in.strip() and pid_in.strip():
            body = claims_in[:limits["max_claim_chars"]]
            df_in = pd.DataFrame([{"Patent ID": pid_in.strip()[:60], "Title": title_in.strip()[:300], "Independent Claims": body}])
            df_key = hashlib.sha256(("paste|" + pid_in + title_in + body).encode()).hexdigest()

    pf = None
    if df_in is not None:
        try:
            pf = build_portfolio(df_in, df_key)
            _, _, n_excl, hits = st.session_state["pf_memo"]
            ui.kpis([("Patents", len(pf.patents)), ("Independent claims", len(pf.claim_records)),
                     ("Claim elements", len(pf.limitation_records)), ("Excluded by status", n_excl)])
            if hits:
                st.warning("Some claim text contains instruction-like wording. It will be treated strictly as data.")
        except UserError as e:
            st.error(str(e))
        except Exception as e:
            st.error(S.safe_error("The portfolio could not be processed.", e))
    elif pf_memo:
        pf = pf_memo[1]

if pf is None:
    ui.footer()
    st.stop()

# ============================================================================ run analysis
EMB_SLUG = re.sub(r"[^A-Za-z0-9]+", "_", cfg["embed_model"])


def review_cache():
    if PERSIST_REVIEWS:
        return E.ReviewCache(S.DATA_DIR / f"reviews_{E.PROMPT_VERSION}.jsonl")
    return st.session_state.setdefault("review_cache", E.ReviewCache(None))


def run_pipeline(pf, sel_products):
    timings, info = {}, {}
    S.audit("run_start", patents=len(pf.patents), limitations=len(pf.limitation_records), products=len(sel_products))
    with st.status("Analysis in progress", expanded=True) as status:
        try:
            t = time.perf_counter()
            if pf.embeddings is None:
                st.write("Preparing claim elements...")
                bar = st.progress(0.0)
                cache = E.EmbeddingCache(EMB_DIR / f"{EMB_SLUG}.npz", cfg["embed_model"])

                def getter():
                    st.write("Initialising analysis engine (first use may take a moment)...")
                    return get_embedder(cfg["embed_model"], corpus, len(corpus.chunks))

                pf.embeddings, info["embed"] = E.embed_cached(getter, pf.texts, cache, lambda i, n: bar.progress(i / n))
                bar.empty()
                if pf.embeddings.shape[1] != corpus.dim:
                    raise RuntimeError("embedding dimension mismatch")
            timings["claim_preparation"] = time.perf_counter() - t

            t = time.perf_counter()
            st.write("Screening the portfolio against product documentation...")
            broad = E.broad_stage(corpus, pf, cfg, sel_products)
            units = E.build_units(broad["selected_pairs"], broad["claim_choice"], pf, cfg)
            bar = st.progress(0.0)
            E.attach_evidence(units, corpus, pf, cfg, lambda i, n: bar.progress(i / n))
            bar.empty()
            timings["screening"] = time.perf_counter() - t

            t = time.perf_counter()
            cache = review_cache()
            for u in units:
                u["cache_key"] = E.make_unit_cache_key(u, pf, cfg)
            review_rows, pending = E.split_cached(units, cache)
            remaining = S.remaining_calls(user, limits)
            allowed = cfg["max_gemini_calls"] if is_admin else min(cfg["max_gemini_calls"], limits["calls_per_run"])
            if remaining is not None:
                allowed = min(allowed, remaining)
            pending, dropped = E.apply_call_cap(pending, allowed)
            if pending:
                st.write(f"Performing detailed technical review ({len(pending):,} units)...")
                bar = st.progress(0.0)
                call = E.make_gemini_caller(api_key, cfg["gemini_model"], cfg["gemini_temperature"], cfg["gemini_retries"])
                E.run_reviews(pending, call, pf, cache, cfg, review_rows,
                              lambda i, n: bar.progress(i / n, text=f"Detailed review {i} of {n}"))
                bar.empty()
                S.consume_calls(user, len(pending))
            timings["detailed_review"] = time.perf_counter() - t

            detailed = E.finalize_reviews(units, review_rows, cfg)
            result = E.summarise(detailed, broad["selected_pairs"], broad["pair_info"], pf, cfg)
            n_err = sum(r["status"] == "REVIEW_ERROR" for r in detailed)
            about = ["iCUERIOUS - Product-to-Patent Mapping", f"Generated: {date.today().isoformat()}",
                     "Technical prioritisation aid based on product documentation. Not a legal opinion on infringement, "
                     "validity or freedom to operate. Review all findings with a qualified patent professional.",
                     "Confidential: prepared for authorised client use only."]
            views = E.client_views(result)
            sheets = {"Ranked Matches": views["pairs"], "By Patent": views["patents"], "Claim Charts": views["chart"]}
            st.session_state["xlsx"] = E.build_excel(sheets, about)
            if is_admin:
                st.session_state["xlsx_admin"] = E.build_excel(
                    {**sheets, "Internal - Pairs": result["pairs"], "Internal - Claims": result["claims"],
                     "Internal - Pre-review": broad["pairs_df"]}, about)
            st.session_state.update(results=result, results_key=(st.session_state["pf_memo"][0], time.time()), n_err=n_err,
                                    n_dropped=dropped, admin_info={"timings": timings, "embed": info.get("embed"),
                                                                   "sim": broad["similarity_distribution"],
                                                                   "pairs": broad["pairs_df"]})
            S.audit("run_done", pairs=len(result["pairs"]), incomplete=n_err, new_reviews=len(pending),
                    seconds=round(sum(timings.values())))
            status.update(label="Analysis complete", state="complete")
        except Exception as e:
            status.update(label="Analysis could not be completed", state="error")
            st.error(S.safe_error("The analysis could not be completed.", e))


with st.container(border=True):
    ui.section("2. Run analysis", "Each patent is compared with the selected products; the strongest candidates receive a detailed element-by-element review.")
    sel_products = st.multiselect("Products to analyse", corpus.products, default=corpus.products)
    agree = st.checkbox("I am authorised to submit this material, and I understand the results are a technical aid and not legal advice.")
    remaining = S.remaining_calls(user, limits)
    if remaining is not None:
        st.caption(f"Detailed-review allowance remaining today: {remaining:,}")
    if not api_key:
        st.warning("Detailed review is temporarily unavailable. Please contact support.")
    label = "Continue analysis" if step == 3 and st.session_state.get("n_err") else "Run analysis"
    if st.button(label, type="primary", disabled=not (agree and api_key and sel_products)):
        run_pipeline(pf, sel_products)
        st.rerun() if st.session_state.get("results") else None

res = st.session_state.get("results")
if not res or st.session_state.get("results_key", [None])[0] != st.session_state["pf_memo"][0]:
    ui.footer()
    st.stop()

# ============================================================================ results
views = E.client_views(res)
pairs_v, patents_v, chart_v = views["pairs"], views["patents"], views["chart"]
with st.container(border=True):
    ui.section("3. Results")
    if st.session_state.get("n_err") or st.session_state.get("n_dropped"):
        st.warning("Some items have not been reviewed yet (service limits or a temporary issue). "
                   "Use **Continue analysis** to complete them; finished reviews are kept.")
    strong = int(pairs_v["Priority"].str.startswith(("P1", "P2")).sum()) if len(pairs_v) else 0
    ui.kpis([("Patents analysed", len(patents_v)), ("Patent-product pairs", len(pairs_v)),
             ("High / strong candidates", strong),
             ("Incomplete reviews", int((pairs_v["Review"] == "Incomplete").sum()) if len(pairs_v) else 0)])

    f1, f2 = st.columns(2)
    prod_f = f1.multiselect("Filter by product", sorted(pairs_v["Product"].unique()) if len(pairs_v) else [])
    prio_f = f2.multiselect("Filter by priority", ["P1 - High priority", "P2 - Strong candidate", "P3 - Secondary", "P4 - Low", "Pending"])
    view = pairs_v
    if prod_f:
        view = view[view["Product"].isin(prod_f)]
    if prio_f:
        view = view[view["Priority"].isin(prio_f)]

    t1, t2, t3 = st.tabs(["Ranked matches", "By patent", "Claim chart"])
    score_cfg = {"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f")}
    with t1:
        st.dataframe(view, hide_index=True, width="stretch", column_config=score_cfg)
    with t2:
        st.dataframe(patents_v, hide_index=True, width="stretch",
                     column_config={"Best score": st.column_config.ProgressColumn("Best score", min_value=0, max_value=100, format="%.0f")})
    with t3:
        if len(view):
            labels = [f"{r['Patent ID']}  |  {r['Product']}  |  {r['Priority']}" for r in view.to_dict("records")]
            pick = st.selectbox("Select a patent-product pair", labels)
            row = view.iloc[labels.index(pick)]
            sub = chart_v[(chart_v["Patent ID"] == row["Patent ID"]) & (chart_v["Product"] == row["Product"])]
            if pd.notna(row["Claim"]):
                sub = sub[sub["Claim"] == row["Claim"]]
            colors = {"Direct": "#2f9e6b", "Partial": "#e8a317", "Related": "#ee7a1a", "None found": "#9aa0aa", "Not completed": "#b3b8c1"}
            for r in sub.sort_values("Element #").to_dict("records"):
                title = re.sub(r"[\[\]\(\)`*_~<>#]", "", str(r["Claim element"]))[:100]
                with st.expander(f"Element {r['Element #']}: {title}"):
                    st.markdown(f'<span class="badge" style="background:{colors.get(r["Correspondence"], "#9aa0aa")}">'
                                f'{html.escape(str(r["Correspondence"]))}</span> &nbsp;<span class="chip">Confidence: '
                                f'<b>{html.escape(str(r["Confidence"]))}</b></span>', unsafe_allow_html=True)
                    st.text("Claim element:\n" + str(r["Claim element"]))
                    st.text("Rationale:\n" + str(r["Rationale"]))
                    if r["Missing / unclear"]:
                        st.text("Missing / unclear:\n" + str(r["Missing / unclear"]))
                    st.text("Product reference: " + str(r["Product reference"]))
                    st.text(str(r["Product passage"]))
        else:
            st.info("No pairs match the current filters.")

    st.download_button("Download Excel report", st.session_state["xlsx"], "product_to_patent_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       on_click=lambda: S.audit("download", kind="client_report"))

# ============================================================================ admin diagnostics
if is_admin and st.session_state.get("admin_info"):
    with st.expander("Administrator diagnostics"):
        ai = st.session_state["admin_info"]
        st.write({k: round(v, 1) for k, v in ai["timings"].items()})
        st.write(ai["embed"])
        st.dataframe(ai["sim"], hide_index=True, width="stretch")
        st.dataframe(ai["pairs"].head(500), hide_index=True, width="stretch")
        st.download_button("Download full internal workbook", st.session_state["xlsx_admin"], "internal_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
ui.footer()
