# iCuerious Product-to-Patent Mapping: client portal

Streamlit app with a neumorphic UI in the iCuerious palette, built on the v5 notebook pipeline.

## Where to host it (Netlify cannot run Streamlit)
Netlify serves static files and short serverless functions. Streamlit needs a long-running Python server with
WebSockets, which Netlify (including its proxy rewrites) does not support. Use:

| Option | Fit |
|---|---|
| **Google Cloud Run** (Dockerfile included) | Best for a client-facing service: private image, Secret Manager, custom domain, autoscaling. Set min instances = 1 to avoid cold starts. |
| Hugging Face Spaces (private, Docker) | Simple, can add GPU. |
| Any VM + Docker behind Nginx / Cloudflare | Full control. |
| Streamlit Community Cloud | Fine for a demo only (about 1 GB RAM, public-repo habits, no private network controls). |

Keep Netlify for the company website / a branded landing page that links to `https://app.yourdomain.com`.

## Files
```
app.py                        UI + workflow
engine.py                     pipeline (retrieval, shortlist, review, scoring, export)
security.py / pwhash.py       login, lockout, quotas, audit log, upload checks
ui.py                         neumorphic theme (assets/logo.svg overrides the built-in logo mark)
make_password_hash.py         creates password hashes for accounts
precompute_patent_embeddings.py   one-off GPU embedding of a portfolio
corpus/                       chunks.json, embeddings.npy, manifest.json   (PRIVATE, not in git)
emb_cache/                    precomputed patent-element embeddings        (PRIVATE, not in git)
```

## Setup
1. `pip install -r requirements.txt`
2. For each account: `python make_password_hash.py`, paste the output into `[auth.users]` (see `secrets.toml.example`).
   Names listed in `admins` get the admin sidebar and diagnostics; everyone else is a client.
3. Set `GEMINI_API_KEY` as a secret / environment variable. It is never shown in, or accepted from, the UI.
4. Put `chunks.json`, `embeddings.npy`, `manifest.json` in `corpus/` (or mount it and set `CORPUS_DIR`).
5. `streamlit run app.py`

Cloud Run example (env vars instead of secrets.toml): `AUTH_USERS_JSON='{"alice":"pbkdf2_sha256$..."}'`,
`ADMIN_USERS=alice`, `GEMINI_API_KEY` from Secret Manager.

## Faster patent embedding on CPU
1. **Precompute once on Colab (GPU):** `python precompute_patent_embeddings.py portfolio.xlsx --out emb_cache`, then ship
   `emb_cache/*.npz` with the app. The app looks up each claim element by text hash; if everything is cached, **no
   embedding model is even loaded**.
2. Anything new is embedded on CPU once (deduplicated, length-sorted batches, ONNX via fastembed) and added to the cache.
3. Embeddings are stored as float16 (about 1.5 KB per element).
The query model must match the one used for `embeddings.npy`; the app verifies this against stored chunks.

## What protects what
- Code, prompts, scoring and API key live on the server. The browser receives rendered widgets only.
- Sign-in: PBKDF2 password hashes, constant-work checks, lockout after 5 failures (5 min), 30-minute idle timeout.
  The service refuses to start unauthenticated if no accounts are configured.
- Clients see conclusions and evidence only: no model names, similarity scores, chunk ids, settings or diagnostics.
  Errors show a reference id; details go to the server log. Client Excel exports contain no internal columns.
- Guardrails: upload type/size/macro/zip-bomb checks, patent/element/claim-length caps, daily and per-run review
  quotas, terms acknowledgement, prompt-injection instruction plus detection, spreadsheet formula-injection
  neutralisation, LLM text rendered as plain text (never HTML), audit log of metadata only (`.data/audit.jsonl`).
- Uploads are processed in memory. Review results are cached server-side in `.data/`; set `PERSIST_REVIEWS=false`
  to keep them in memory only.

## Still your responsibility
- **Private repository / private image.** Anyone who can read the repo can read the code and prompts.
- Serve over HTTPS on your own domain; add security headers (CSP, HSTS) at the proxy/CDN, which Streamlit cannot set.
- Claim text and product passages are sent to Google's Gemini API. Use a paid tier whose terms exclude training on
  your data, and tell clients so in your contract.
- Per-user lockout means someone who knows a username can lock it out for 5 minutes; use an IdP/SSO for stricter needs.
- Back up / rotate `.data/` and the API key; review the audit log.
