"""Evaluate a saved score file on val. Usage:
python my_models/eval_scores.py item2vec            # target file (val rows)
python my_models/eval_scores.py item2vec stack      # stack file (needs labels, see note)
"""
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics import evaluate  # noqa: E402

name = sys.argv[1]
kind = sys.argv[2] if len(sys.argv) > 2 else "target"
path = ROOT / "data" / "scores" / "val" / f"{name}_{kind}.parquet"
preds = pl.read_parquet(path).rename({f"{name}_score": "predicted_probability"})
print(f"{name} {kind}: {preds.height:,} rows")
evaluate(preds)
