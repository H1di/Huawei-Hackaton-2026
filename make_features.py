"""Признаки Назара (features.build) для бустинга, но кусками — чтобы влезть в память.

    python make_features.py --mode val      # fit_features (stack-окно) + val_features
    python make_features.py --mode final    # train_features (stack-окно) + eval_features

Почему отдельный скрипт, а не `python features.py`: тот грузит все 4 таблицы целиком
со списками истории и разворачивает историю построчно (fit: 4.9M x 20 строк) —
на 14 GB это убивается ядром (OOM).

Почему результат тот же самый:
  * prior_stats для строки зависит только от самой строки и history -> строки можно
    гнать любыми кусками, history (без списков) лежит в памяти целиком;
  * ctr_vs_event_mean / ctr_rank_in_event считаются внутри события -> режем по событиям,
    все 10 строк события всегда в одном куске.
Для fit/train считаем только stack-окно: бустинг учится только на нём (см. booster.py).
"""
import argparse

import polars as pl

import config as C
from features import build

OUT = C.ROOT / "data" / "features"
HIST_COLS = ["user_id", "item_id", "timestamp", "event_id", "clicked"]   # prior_stats больше не нужно
EVENTS_PER_CHUNK = 50_000                                                # ~500k строк на кусок


def build_chunked(rows: pl.LazyFrame, history: pl.DataFrame, out_name: str) -> None:
    events = rows.select("event_id").unique().sort("event_id").collect()["event_id"]
    n_chunks = max(1, -(-len(events) // EVENTS_PER_CHUNK))
    parts = []
    for i in range(n_chunks):
        ev = events[i * EVENTS_PER_CHUNK:(i + 1) * EVENTS_PER_CHUNK].implode()
        chunk = rows.filter(pl.col("event_id").is_in(ev)).collect()
        parts.append(build(chunk, history))
        print(f"  {out_name}: кусок {i + 1}/{n_chunks}, {chunk.height:,} строк", flush=True)
    df = pl.concat(parts)
    assert df["rowid"].n_unique() == df.height == rows.select(pl.len()).collect().item()
    df.write_parquet(OUT / out_name)
    print(f"готово: {OUT / out_name} ({df.height:,} строк, {df.width} колонок)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "final"], required=True)
    final = ap.parse_args().mode == "final"
    part_pq, target_pq = (C.TRAIN_PQ, C.EVAL_PQ) if final else (C.FIT_PQ, C.VAL_PQ)
    w = C.windows(final=final)
    OUT.mkdir(parents=True, exist_ok=True)

    history = pl.read_parquet(part_pq, columns=HIST_COLS)
    stack = pl.scan_parquet(part_pq).filter(pl.col("timestamp") >= w["stack_start"],
                                            pl.col("timestamp") < w["part_end"])
    build_chunked(stack, history, "train_features.parquet" if final else "fit_features.parquet")
    build_chunked(pl.scan_parquet(target_pq), history, "eval_features.parquet" if final else "val_features.parquet")


if __name__ == "__main__":
    main()
