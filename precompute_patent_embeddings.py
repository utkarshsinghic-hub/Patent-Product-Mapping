"""Run ONCE on a GPU machine (e.g. Colab) to embed a patent portfolio, so the CPU host never has to.

    python precompute_patent_embeddings.py portfolio.xlsx --model BAAI/bge-base-en-v1.5 --out emb_cache

Copy the resulting emb_cache/<model>.npz next to app.py (or set EMB_CACHE_DIR). The app looks up every claim
element by text hash, so new or changed patents are embedded on the fly and only those, then cached.
Re-running this script on an updated portfolio only embeds what is new.
"""
import argparse
import re
from pathlib import Path

import engine as E

ap = argparse.ArgumentParser()
ap.add_argument("portfolio")
ap.add_argument("--model", default=E.DEFAULTS["embed_model"])
ap.add_argument("--out", default="emb_cache")
ap.add_argument("--batch", type=int, default=256)
a = ap.parse_args()

df = E.read_portfolio_file(a.portfolio, Path(a.portfolio).read_bytes())
cfg = dict(E.DEFAULTS)
pf = E.Portfolio(df, E.resolve_columns(df, cfg))        # same parser as the app, so the texts hash identically
print(f"{len(pf.patents):,} patents, {len(pf.texts):,} claim elements")

cache = E.EmbeddingCache(Path(a.out) / f"{re.sub(r'[^A-Za-z0-9]+', '_', a.model)}.npz", a.model)
_, stats = E.embed_cached(lambda: E.Embedder(a.model), pf.texts, cache, batch=a.batch,
                          on_progress=lambda i, n: print(f"  embedded {i:,}/{n:,}", end="\r"))
print(f"\n{stats}\nSaved to {cache.path}")
