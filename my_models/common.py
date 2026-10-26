"""Event-level tables, item vocab, loaders, and score I/O.

Work at event granularity so history is stored once per event, not copied
across the 10 candidate rows.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as C

SEED = 0
PAD_ID = 0
UNK_ID = 1
MAX_HIST = 20
N_CAND = C.CANDIDATES_PER_EVENT
Mode = Literal["val", "final"]
Kind = Literal["stack", "target"]

# History timestamps / extra feature cols stay on disk: they are unused here.
_ROW_COLS = ["rowid", "event_id", "user_id", "item_id", "timestamp", "history_item_ids"]


@dataclass
class EventTable:
    """One record per event: shared history + 10 candidates."""

    event_id: pl.Series
    user_id: pl.Series
    timestamp: pl.Series
    history: pl.Series  # List[str], oldest -> newest, may be empty
    item_ids: pl.Series  # List[str], length 10
    rowids: pl.Series  # List[str], length 10, opaque TEXT
    labels: pl.Series | None = None  # List[i8] length 10, or None (evaluation)

    def __len__(self) -> int:
        return int(self.event_id.len())

    def n_rows(self) -> int:
        return len(self) * N_CAND

    def to_frame(self) -> pl.DataFrame:
        cols: dict[str, pl.Series] = {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "history": self.history,
            "item_ids": self.item_ids,
            "rowids": self.rowids,
        }
        if self.labels is not None:
            cols["labels"] = self.labels
        return pl.DataFrame(cols)

    @classmethod
    def from_event_df(cls, df: pl.DataFrame) -> EventTable:
        """Build from an already-aggregated event-level frame."""
        need = {"event_id", "user_id", "timestamp", "history", "item_ids", "rowids"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"event-level frame missing {missing}")
        n = df["item_ids"].list.len()
        if df.height and not bool((n == N_CAND).all()):
            raise ValueError(f"each event must have {N_CAND} candidates")
        n_r = df["rowids"].list.len()
        if df.height and not bool((n_r == N_CAND).all()):
            raise ValueError(f"each event must have {N_CAND} rowids")
        labels = df["labels"] if "labels" in df.columns else None
        if labels is not None and df.height and not bool((labels.list.len() == N_CAND).all()):
            raise ValueError(f"each event must have {N_CAND} labels")
        return cls(
            event_id=df["event_id"],
            user_id=df["user_id"],
            timestamp=df["timestamp"],
            history=df["history"],
            item_ids=df["item_ids"],
            rowids=df["rowids"],
            labels=labels,
        )

    @classmethod
    def from_df(cls, df: pl.DataFrame) -> EventTable:
        """Accept row-level (item_id) or event-level (item_ids) frames."""
        if "item_ids" in df.columns:
            return cls.from_event_df(df)
        return cls.from_event_df(_rows_to_events(df))

    def flat_rowids(self) -> pl.Series:
        return self.rowids.explode().alias("rowid")


class Vocab:
    """item_id -> int. 0=PAD, 1=UNK. Fit only on the model's training events."""

    PAD = PAD_ID
    UNK = UNK_ID

    def __init__(self, item_to_id: dict[str, int]):
        self.item_to_id = item_to_id

    def __len__(self) -> int:
        return len(self.item_to_id) + 2

    @classmethod
    def fit(cls, events: EventTable) -> Vocab:
        # Candidates and histories of the fitting part only (CONTEXT).
        frame = events.to_frame()
        if frame.height == 0:
            return cls({})
        items = (
            pl.concat([
                frame.select(pl.col("item_ids").explode().alias("item_id")),
                frame.select(pl.col("history").explode().alias("item_id")),
            ])
            .drop_nulls()
            .filter(pl.col("item_id") != "")
            .unique()
            .sort("item_id")
        )
        mapping = {item: i + 2 for i, item in enumerate(items["item_id"].to_list())}
        return cls(mapping)

    def encode(self, item_id: str | None) -> int:
        if item_id is None or item_id == "":
            return self.UNK
        return self.item_to_id.get(item_id, self.UNK)

    def encode_seq(self, items: list[str] | None) -> list[int]:
        if not items:
            return []
        return [self.encode(x) for x in items]


def _scan_needed(path: Path) -> pl.LazyFrame:
    schema_names = pl.scan_parquet(path).collect_schema().names()
    cols = [c for c in _ROW_COLS if c in schema_names]
    if "clicked" in schema_names:
        cols.append("clicked")
    # Projection after scan: polars 2.x scan_parquet has no columns= kwarg.
    return pl.scan_parquet(path).select(cols)


def _sample_whole_events(lf: pl.LazyFrame, fraction: float, seed: int = SEED) -> pl.LazyFrame:
    """Drop or keep every candidate of an event together."""
    if fraction >= 1.0:
        return lf
    if fraction <= 0.0:
        raise ValueError("sample fraction must be > 0")
    ids = lf.select("event_id").unique().collect()
    n = max(1, int(ids.height * fraction))
    pick = ids.sample(n=n, seed=seed, shuffle=True)
    return lf.filter(pl.col("event_id").is_in(pick.get_column("event_id")))


def _rows_to_events(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse 10 candidate rows -> 1 event. history taken once (first)."""
    return _event_agg(df.lazy()).collect()


def _event_agg(lf: pl.LazyFrame) -> pl.LazyFrame:
    names = lf.collect_schema().names()
    aggs: list[pl.Expr] = [
        pl.col("user_id").first(),
        pl.col("timestamp").first(),
        # first() so the list column is not repeated 10x in the result.
        pl.col("history_item_ids").first().alias("history"),
        pl.col("item_id").alias("item_ids"),
        pl.col("rowid").alias("rowids"),
    ]
    if "clicked" in names:
        aggs.append(pl.col("clicked").alias("labels"))
    return lf.group_by("event_id").agg(aggs)


def _read_event_frame(path: Path, sample: float = 1.0) -> pl.DataFrame:
    lf = _sample_whole_events(_scan_needed(path), sample)
    return _event_agg(lf).collect()


def load_part(
    mode: str,
    sample: float = 1.0,
    *,
    sample_target: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Event-level frames for the fitting part and the target, via config paths.

    ``sample`` draws a fraction of whole events from the part. Target is left
    complete unless ``sample_target`` is true, so ``metrics.evaluate`` can run.
    """
    if mode not in {"val", "final"}:
        raise ValueError("mode must be 'val' or 'final'")
    if mode == "val":
        part_path, target_path = C.FIT_PQ, C.VAL_PQ
    else:
        part_path, target_path = C.TRAIN_PQ, C.EVAL_PQ
    part = _read_event_frame(part_path, sample)
    target = _read_event_frame(target_path, sample if sample_target else 1.0)
    return part, target


def load_windows(mode: str) -> dict:
    """Time cuts from config.windows; never hardcode dates."""
    if mode not in {"val", "final"}:
        raise ValueError("mode must be 'val' or 'final'")
    return C.windows(final=(mode == "final"))


def split_by_time(
    events: EventTable,
    start: datetime | None,
    end: datetime | None,
) -> EventTable:
    """Keep events with start <= timestamp < end. None means an open side."""
    df = events.to_frame()
    pred = pl.lit(True)
    if start is not None:
        pred = pred & (pl.col("timestamp") >= start)
    if end is not None:
        pred = pred & (pl.col("timestamp") < end)
    return EventTable.from_event_df(df.filter(pred))


def save_scores(
    df: pl.DataFrame,
    mode: str,
    name: str,
    kind: str,
    expected_rows: int | None = None,
) -> Path:
    """Write data/scores/<mode>/<name>_<kind>.parquet after CONTEXT asserts."""
    if mode not in {"val", "final"}:
        raise ValueError("mode must be 'val' or 'final'")
    if kind not in {"stack", "target"}:
        raise ValueError("kind must be 'stack' or 'target'")
    score_col = f"{name}_score"
    if "rowid" not in df.columns or score_col not in df.columns:
        raise ValueError(f"need columns rowid and {score_col}, got {df.columns}")
    out = df.select(pl.col("rowid"), pl.col(score_col).cast(pl.Float64))
    if out["rowid"].n_unique() != out.height:
        raise ValueError("rowid is not unique")
    s = out[score_col]
    if s.null_count() or bool(s.is_nan().any()) or bool(s.is_infinite().any()):
        raise ValueError("scores must be finite (no null/NaN/inf)")
    if expected_rows is not None and out.height != expected_rows:
        raise ValueError(f"row count {out.height} != expected {expected_rows}")
    dest_dir = C.ROOT / "data" / "scores" / mode
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{name}_{kind}.parquet"
    out.write_parquet(path)
    return path
