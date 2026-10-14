"""Neumorphic theme and small HTML components. Every dynamic value is escaped."""
from __future__ import annotations

import html
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

BRAND = "iCUERIOUS"
TAGLINE = "Finding What Matters Most"
PRODUCT = "Product-to-Patent Mapping"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
:root{
  --bg:#e6e9ef; --sh-d:#c2c7d1; --sh-l:#ffffff; --accent:#ee3b1a; --accent-d:#cf2f10;
  --ink:#3b3f47; --muted:#7a808b; --brown:#46363a; --ok:#2f9e6b; --warn:#e8a317;
  --raise: 8px 8px 18px var(--sh-d), -8px -8px 18px var(--sh-l);
  --raise-s: 5px 5px 11px var(--sh-d), -5px -5px 11px var(--sh-l);
  --inset: inset 5px 5px 10px var(--sh-d), inset -5px -5px 10px var(--sh-l);
}
html, body, .stApp, [class*="css"], button, input, textarea{font-family:'Roboto','Segoe UI',Arial,sans-serif !important;}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]{background:var(--bg) !important; color:var(--ink);}
[data-testid="stHeader"]{background:transparent !important; height:0;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stDeployButton"], .stDeployButton{display:none !important; visibility:hidden !important;}
.block-container{max-width:1240px; padding:1.4rem 2rem 4rem;}
h1,h2,h3,h4{color:var(--ink); font-weight:500; letter-spacing:.2px;}
h2{font-size:1.25rem !important; margin-top:.4rem;}

/* ---------- cards ---------- */
.neo{background:var(--bg); border-radius:22px; box-shadow:var(--raise); padding:1.3rem 1.6rem; margin:0 0 1.3rem;}
.neo.flat{box-shadow:var(--raise-s); padding:1rem 1.3rem;}
.neo.inset{box-shadow:var(--inset);}
.brandbar{display:flex; align-items:center; gap:1.1rem;}
.brandbar .word{font-size:1.7rem; font-weight:500; color:var(--accent); letter-spacing:1.5px; line-height:1;}
.brandbar .word sup{font-size:.45rem; color:var(--muted); margin-left:2px;}
.brandbar .tag{font-size:.78rem; color:var(--ink); margin-top:.35rem; border-top:1px solid var(--sh-d); padding-top:.3rem;}
.brandbar .sep{width:1px; height:44px; background:var(--sh-d); margin:0 .6rem;}
.brandbar .prod{font-size:1.05rem; color:var(--ink); font-weight:500;}
.brandbar .sub{font-size:.8rem; color:var(--muted);}
.chip{display:inline-block; padding:.28rem .8rem; border-radius:999px; box-shadow:var(--inset); font-size:.78rem; color:var(--muted);}
.chip b{color:var(--ink); font-weight:500;}
.kpis{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:1.2rem; margin:.2rem 0 1.4rem;}
.kpi{background:var(--bg); border-radius:18px; box-shadow:var(--raise-s); padding:1rem 1.2rem;}
.kpi .v{font-size:1.9rem; font-weight:300; color:var(--accent); line-height:1.1;}
.kpi .l{font-size:.74rem; color:var(--muted); text-transform:uppercase; letter-spacing:.9px; margin-top:.25rem;}
.stepper{display:flex; gap:1.2rem; margin:.2rem 0 1.5rem; flex-wrap:wrap;}
.step{display:flex; align-items:center; gap:.7rem; padding:.55rem 1.1rem; border-radius:999px; font-size:.86rem; color:var(--muted); box-shadow:var(--inset);}
.step .n{width:26px; height:26px; border-radius:50%; display:grid; place-items:center; font-weight:500; background:var(--bg); box-shadow:var(--raise-s); color:var(--muted); font-size:.8rem;}
.step.on{color:var(--ink); box-shadow:var(--raise-s);} .step.on .n{background:var(--accent); color:#fff; box-shadow:3px 3px 7px var(--sh-d), -3px -3px 7px var(--sh-l);}
.step.done .n{color:var(--ok);}
.badge{display:inline-block; padding:.12rem .6rem; border-radius:8px; font-size:.74rem; font-weight:500; color:#fff;}
.b-P1{background:#d62f0f;} .b-P2{background:#ee7a1a;} .b-P3{background:#e8a317;} .b-P4{background:#9aa0aa;}
.disclaimer{font-size:.76rem; color:var(--muted); line-height:1.5; text-align:center; margin-top:2.2rem;}

[data-testid="stVerticalBlockBorderWrapper"]{border:none !important; border-radius:22px !important; box-shadow:var(--raise); background:var(--bg); padding:1.2rem 1.5rem; margin-bottom:1.4rem;}

/* ---------- buttons ---------- */
.stButton>button, .stDownloadButton>button, [data-testid="stFormSubmitButton"]>button{
  background:var(--bg) !important; color:var(--accent) !important; border:none !important; border-radius:14px !important;
  font-weight:500 !important; padding:.6rem 1.5rem !important; box-shadow:var(--raise-s) !important; transition:all .15s ease;}
.stButton>button:hover, .stDownloadButton>button:hover{color:var(--accent-d) !important; transform:translateY(-1px);}
.stButton>button:active, .stDownloadButton>button:active{box-shadow:var(--inset) !important; transform:none;}
.stButton>button[kind="primary"], [data-testid="stFormSubmitButton"]>button[kind="primaryFormSubmit"], [data-testid="stFormSubmitButton"]>button[kind="primary"]{
  background:linear-gradient(145deg,#f4502f,#e03314) !important; color:#fff !important;
  box-shadow:6px 6px 14px #c9a39b, -6px -6px 14px #fff !important;}
.stButton>button[kind="primary"]:active{box-shadow:inset 4px 4px 9px #b8280d, inset -4px -4px 9px #f66a4d !important;}
.stButton>button:disabled{color:#b3b8c1 !important; box-shadow:var(--raise-s) !important; opacity:.8;}

/* ---------- inputs ---------- */
.stTextInput input, .stTextArea textarea, .stNumberInput input, [data-baseweb="select"]>div, [data-baseweb="input"]{
  background:var(--bg) !important; border:none !important; border-radius:12px !important; box-shadow:var(--inset) !important; color:var(--ink) !important;}
[data-baseweb="base-input"]{background:transparent !important;}
.stTextInput label, .stTextArea label, .stSelectbox label, .stMultiSelect label, .stFileUploader label, .stCheckbox label{color:var(--ink) !important; font-size:.86rem;}
[data-baseweb="tag"]{background:var(--accent) !important; border-radius:8px !important;}
[data-testid="stFileUploaderDropzone"]{background:var(--bg) !important; border:none !important; border-radius:18px !important; box-shadow:var(--inset) !important; padding:1.6rem !important;}
[data-testid="stFileUploaderDropzone"] button{box-shadow:var(--raise-s) !important;}

/* ---------- tabs / expanders / tables / progress ---------- */
[data-baseweb="tab-list"]{gap:.6rem; background:transparent !important; border:none !important; padding:.4rem .2rem 1rem;}
button[data-baseweb="tab"]{background:var(--bg) !important; border-radius:12px !important; box-shadow:var(--raise-s); color:var(--muted) !important; padding:.5rem 1.2rem !important; height:auto !important;}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--accent) !important; box-shadow:var(--inset);}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{display:none !important;}
[data-testid="stExpander"]{background:var(--bg); border:none !important; border-radius:16px; box-shadow:var(--raise-s); margin-bottom:.9rem;}
[data-testid="stExpander"] details{border:none !important;} [data-testid="stExpander"] summary{color:var(--ink);}
[data-testid="stDataFrame"]{border-radius:16px; box-shadow:var(--raise-s); padding:6px; background:var(--bg); overflow:hidden;}
[data-testid="stProgress"] div[role="progressbar"]>div{background:linear-gradient(90deg,#f4502f,#ee3b1a) !important;}
[data-testid="stProgress"] div[role="progressbar"]{background:var(--bg) !important; box-shadow:var(--inset); border-radius:999px;}
[data-testid="stStatus"], [data-testid="stAlert"]{border-radius:16px !important; border:none !important; box-shadow:var(--raise-s); background:var(--bg) !important;}
[data-testid="stMetric"]{background:var(--bg); border-radius:16px; box-shadow:var(--raise-s); padding:.9rem 1.1rem;}
hr{border:none; height:1px; background:var(--sh-d); opacity:.5;}
::-webkit-scrollbar{width:10px; height:10px;} ::-webkit-scrollbar-thumb{background:var(--sh-d); border-radius:8px;}
</style>
"""

HIDE_SIDEBAR = """<style>[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"]{display:none !important;}</style>"""


def inject_theme(hide_sidebar: bool = False):
    st.markdown(CSS, unsafe_allow_html=True)
    if hide_sidebar:
        st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)


def logo_svg(h: int = 54) -> str:
    p = Path(__file__).parent / "assets" / "logo.svg"      # drop your official logo here to replace the approximation
    if p.exists():
        return f'<div style="height:{h}px;display:flex;align-items:center">{p.read_text(encoding="utf-8")}</div>'
    return (f'<svg height="{h}" viewBox="0 0 34 46" xmlns="http://www.w3.org/2000/svg" aria-label="logo">'
            '<circle cx="17" cy="9" r="9" fill="#ee3b1a"/>'
            '<path d="M1 22h32v14a8 8 0 0 1-8 8H9a8 8 0 0 1-8-8z" fill="#ee3b1a"/></svg>')


def _brand(sub: str = "") -> str:
    return (f'<div class="brandbar">{logo_svg()}<div><div class="word">{BRAND}<sup>TM</sup></div>'
            f'<div class="tag">{TAGLINE}</div></div><div class="sep"></div>'
            f'<div><div class="prod">{PRODUCT}</div><div class="sub">{html.escape(sub)}</div></div></div>')


def header(user_name: str, role: str):
    c1, c2 = st.columns([5, 1.4], vertical_alignment="center")
    with c1:
        st.markdown(f'<div class="neo">{_brand("Technical correspondence analysis for patent portfolios")}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div style="text-align:right;margin-bottom:.5rem"><span class="chip">Signed in as <b>{html.escape(user_name)}</b>'
                    f'{" &middot; admin" if role == "admin" else ""}</span></div>', unsafe_allow_html=True)
        return st.button("Sign out", use_container_width=True)


def login_card(notice: Optional[str]):
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown('<div style="height:7vh"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="neo" style="text-align:center">{_brand("Secure client portal")}'
                    '<p style="color:var(--muted);font-size:.85rem;margin:1.1rem 0 0">'
                    'Authorised users only. Activity is logged.</p></div>', unsafe_allow_html=True)
        if notice:
            st.info(notice)


def stepper(current: int, labels: Tuple[str, ...] = ("Upload portfolio", "Run analysis", "Review results")):
    parts = []
    for i, lab in enumerate(labels, start=1):
        cls = "on" if i == current else ("done" if i < current else "")
        mark = "&#10003;" if i < current else str(i)
        parts.append(f'<div class="step {cls}"><span class="n">{mark}</span>{html.escape(lab)}</div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def kpis(items: List[Tuple[str, object]]):
    cells = "".join(f'<div class="kpi"><div class="v">{html.escape(str(v))}</div><div class="l">{html.escape(l)}</div></div>'
                    for l, v in items)
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


def badge(code: str) -> str:
    code = code if code in ("P1", "P2", "P3", "P4") else "P4"
    return f'<span class="badge b-{code}">{code}</span>'


def section(title: str, caption: str = ""):
    st.markdown(f"## {html.escape(title)}")
    if caption:
        st.caption(caption)


def footer():
    st.markdown('<div class="disclaimer">Results are a technical prioritisation aid based on product documentation. '
                'They are not a legal opinion on infringement, validity or freedom to operate. '
                'Review all findings with a qualified patent professional.<br>'
                f'&copy; {BRAND}. Confidential &mdash; for authorised client use only.</div>', unsafe_allow_html=True)
