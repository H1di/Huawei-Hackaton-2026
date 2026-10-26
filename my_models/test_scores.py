"""Scaffolding checks. Synthetic events only — no real parquet rows.

Run: python my_models/test_scores.py
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import (  # noqa: E402
    N_CAND,
    UNK_ID,
    EventTable,
    Vocab,
    load_windows,
    save_scores,
    split_by_time,
)
from dataset import to_tensors  # noqa: E402
from run_scores import PopularityModel  # noqa: E402


def _event(
    event_id: str,
    ts: datetime,
    history: list[str],
    items: list[str],
    labels: list[int] | None,
    user_id: str = "u",
) -> dict:
    assert len(items) == N_CAND
    rowids = [f"{event_id}_{k}" for k in range(N_CAND)]
    rec = {
        "event_id": event_id,
        "user_id": user_id,
        "timestamp": ts,
        "history": history,
        "item_ids": items,
        "rowids": rowids,
    }
    if labels is not None:
        rec["labels"] = labels
    return rec


def _table(rows: list[dict]) -> EventTable:
    return EventTable.from_event_df(pl.DataFrame(rows))


def _fit_base(ts: datetime) -> EventTable:
    """Two events covering items a..t so CTR is identified."""
    items_1 = [f"i{k}" for k in range(N_CAND)]
    items_2 = [f"j{k}" for k in range(N_CAND)]
    y1 = [0] * N_CAND
    y1[0] = 1
    y2 = [0] * N_CAND
    return _table([
        _event("e1", ts, ["h0", "h1"], items_1, y1),
        _event("e2", ts + timedelta(seconds=1), [], items_2, y2),
    ])


def test_empty_history_finite_score():
    ts = datetime(2025, 1, 20, tzinfo=timezone.utc)
    train = _fit_base(ts)
    # Scoring event with no history still gets item popularity, not NaN.
    empty = _table([_event(
        "empty", ts + timedelta(days=1),
        history=[],
        items=[f"i{k}" for k in range(N_CAND)],
        labels=None,
    )])
    m = PopularityModel()
    m.fit(train)
    scores = m.score(empty)["popularity_score"]
    assert scores.null_count() == 0
    assert not bool(scores.is_nan().any())
    assert not bool(scores.is_infinite().any())

    vocab = Vocab.fit(train)
    batch = to_tensors(empty, vocab)
    assert batch.hist_mask.sum().item() == 0
    assert int(batch.history.sum()) == 0


def test_unknown_item_is_unk_and_global_ctr():
    ts = datetime(2025, 1, 20, tzinfo=timezone.utc)
    train = _fit_base(ts)
    m = PopularityModel()
    m.fit(train)
    vocab = Vocab.fit(train)
    assert vocab.encode("not-in-fit") == UNK_ID

    unseen = [f"z{k}" for k in range(N_CAND)]
    ev = _table([_event("u1", ts, ["also-new"], unseen, None)])
    scores = m.score(ev)["popularity_score"].to_list()
    assert all(math.isclose(s, m.global_ctr, rel_tol=0, abs_tol=1e-12) for s in scores)

    batch = to_tensors(ev, vocab)
    assert set(batch.cand.unique().tolist()) == {UNK_ID}
    assert int(batch.history[0, -1].item()) == UNK_ID


def test_rowids_in_equal_rowids_out():
    ts = datetime(2025, 1, 20, tzinfo=timezone.utc)
    train = _fit_base(ts)
    m = PopularityModel()
    m.fit(train)
    got = m.score(train)["rowid"].to_list()
    want = train.flat_rowids().to_list()
    assert got == want


def test_stack_window_never_overlaps_training_part():
    w = load_windows("val")
    assert w["stack_start"] < w["part_end"]
    items = [f"i{k}" for k in range(N_CAND)]
    y = [0] * N_CAND
    rows = [
        _event("before", w["stack_start"] - timedelta(seconds=1), [], items, y),
        _event("at_stack", w["stack_start"], [], items, y),
        _event("in_stack", w["stack_start"] + timedelta(hours=1), [], items, y),
        _event("at_end", w["part_end"], [], items, y),
    ]
    events = _table(rows)
    train = split_by_time(events, start=None, end=w["stack_start"])
    stack = split_by_time(events, start=w["stack_start"], end=w["part_end"])
    assert set(train.event_id.to_list()) == {"before"}
    assert set(stack.event_id.to_list()) == {"at_stack", "in_stack"}
    assert train.timestamp.max() < stack.timestamp.min()
    assert set(train.event_id.to_list()).isdisjoint(set(stack.event_id.to_list()))

    # Same cut on final windows.
    wf = load_windows("final")
    train_f = split_by_time(events, start=None, end=wf["stack_start"])
    stack_f = split_by_time(events, start=wf["stack_start"], end=wf["part_end"])
    if len(train_f) and len(stack_f):
        assert train_f.timestamp.max() < stack_f.timestamp.min()


def test_no_nan_inf_in_scores():
    ts = datetime(2025, 1, 20, tzinfo=timezone.utc)
    train = _fit_base(ts)
    m = PopularityModel()
    m.fit(train)
    s = m.score(train)["popularity_score"]
    arr = s.to_numpy()
    assert np.isfinite(arr).all()
    path = save_scores(m.score(train), "val", "popularity", "stack", expected_rows=train.n_rows())
    assert path.exists()


def test_per_event_shape_is_10():
    ts = datetime(2025, 1, 20, tzinfo=timezone.utc)
    train = _fit_base(ts)
    assert len(train) == 2
    assert train.n_rows() == 2 * N_CAND
    assert (train.item_ids.list.len() == N_CAND).all()
    assert (train.rowids.list.len() == N_CAND).all()
    vocab = Vocab.fit(train)
    batch = to_tensors(train, vocab)
    assert tuple(batch.history.shape) == (2, 20)
    assert tuple(batch.hist_mask.shape) == (2, 20)
    assert tuple(batch.cand.shape) == (2, N_CAND)
    assert tuple(batch.label.shape) == (2, N_CAND)
    assert batch.rowid.shape == (2, N_CAND)
    # Left pad: known history occupies the tail.
    assert bool(batch.hist_mask[0, -2:].all())
    assert not bool(batch.hist_mask[0, :-2].any())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("Все проверки прошли")
