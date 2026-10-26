"""Two-stage scores: stack window for the booster, then a fresh fit for the target.

A model never scores rows it was trained on. Vocab is rebuilt from the fitting
events of that stage only.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import (  # noqa: E402
    EventTable,
    Vocab,
    load_part,
    load_windows,
    save_scores,
    split_by_time,
)

try:
    import torch

    torch.set_num_threads(16)
except ImportError:
    pass


@dataclass
class PopularityConfig:
    """Laplace-smoothed item CTR. Alpha=1 -> (c+1)/(n+2), prior at 0.5."""

    alpha: float = 1.0
    seed: int = 0


class PopularityModel:
    """Per-item CTR on the fitting part; items never seen as candidates -> global CTR."""

    name = "popularity"

    def __init__(self, cfg: PopularityConfig | None = None):
        self.cfg = cfg or PopularityConfig()
        self.vocab: Vocab | None = None
        self.item_ctr: dict[str, float] = {}
        self.global_ctr: float = 0.5

    def fit(self, events: EventTable) -> None:
        self.vocab = Vocab.fit(events)
        a = self.cfg.alpha
        if events.labels is None or len(events) == 0:
            self.item_ctr = {}
            self.global_ctr = 0.5
            return
        rows = events.to_frame().select(
            pl.col("item_ids").explode().alias("item_id"),
            pl.col("labels").explode().alias("clicked"),
        )
        n_all = rows.height
        c_all = float(rows["clicked"].sum())
        self.global_ctr = (c_all + a) / (n_all + 2.0 * a)
        stats = rows.group_by("item_id").agg(
            pl.len().alias("n"),
            pl.col("clicked").cast(pl.Float64).sum().alias("c"),
        ).with_columns(
            ((pl.col("c") + a) / (pl.col("n") + 2.0 * a)).alias("ctr")
        )
        self.item_ctr = dict(zip(stats["item_id"].to_list(), stats["ctr"].to_list()))

    def score(self, events: EventTable) -> pl.DataFrame:
        g = self.global_ctr
        rows = events.to_frame().select(
            pl.col("rowids").explode().alias("rowid"),
            pl.col("item_ids").explode().alias("item_id"),
        )
        if not self.item_ctr:
            return rows.select(
                "rowid",
                pl.lit(g).cast(pl.Float64).alias(f"{self.name}_score"),
            )
        ctr = pl.DataFrame({
            "item_id": list(self.item_ctr.keys()),
            f"{self.name}_score": list(self.item_ctr.values()),
        })
        return (
            rows.join(ctr, on="item_id", how="left")
            .with_columns(pl.col(f"{self.name}_score").fill_null(g))
            .select("rowid", f"{self.name}_score")
        )


class InHistModel:
    """1.0 if the candidate appears in this event's history, else 0.0.

    fit() still builds a vocab from the fitting part so the two-stage CLI
    stays uniform; the score itself does not use that vocab.
    """

    name = "inhist"

    def __init__(self):
        self.vocab: Vocab | None = None

    def fit(self, events: EventTable) -> None:
        self.vocab = Vocab.fit(events)

    def score(self, events: EventTable) -> pl.DataFrame:
        col = f"{self.name}_score"
        return (
            events.to_frame()
            .select("rowids", "item_ids", "history")
            .explode("rowids", "item_ids")
            .select(
                pl.col("rowids").alias("rowid"),
                pl.col("history")
                .list.contains(pl.col("item_ids"))
                .cast(pl.Float64)
                .alias(col),
            )
        )


MODELS: dict[str, type] = {
    "popularity": PopularityModel,
    "inhist": InHistModel,
}


def run(model_name: str, mode: str, sample: float) -> tuple[Path, Path]:
    if model_name not in MODELS:
        raise SystemExit(f"unknown model {model_name!r}, have {sorted(MODELS)}")
    w = load_windows(mode)
    part_df, target_df = load_part(mode, sample=sample, sample_target=False)
    part = EventTable.from_df(part_df)
    target = EventTable.from_df(target_df)
    del part_df, target_df

    train_a = split_by_time(part, start=None, end=w["stack_start"])
    stack = split_by_time(part, start=w["stack_start"], end=w["part_end"])
    if len(train_a) == 0:
        raise RuntimeError("training slice before stack_start is empty")
    if len(stack) == 0:
        raise RuntimeError("stack window is empty (try a larger --sample)")

    model_a = MODELS[model_name]()
    model_a.fit(train_a)
    stack_scores = model_a.score(stack)
    stack_path = save_scores(
        stack_scores, mode, model_name, "stack", expected_rows=stack.n_rows()
    )

    # Fresh model: vocab and weights come from the full part, never from the target.
    model_b = MODELS[model_name]()
    model_b.fit(part)
    target_scores = model_b.score(target)
    target_path = save_scores(
        target_scores, mode, model_name, "target", expected_rows=target.n_rows()
    )
    print(f"wrote {stack_path} ({stack_scores.height} rows)")
    print(f"wrote {target_path} ({target_scores.height} rows)")
    return stack_path, target_path


def main() -> None:
    p = argparse.ArgumentParser(description="Fit a my_models scorer and write stack/target parquet.")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--mode", required=True, choices=("val", "final"))
    p.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="fraction of PART events (whole events). Target is scored in full.",
    )
    args = p.parse_args()
    run(args.model, args.mode, args.sample)


if __name__ == "__main__":
    main()
