"""Шаг 1. CSV -> Parquet (polars, стриминг), JSON-истории сразу парсятся в списки.

Запуск:
    python data_prep.py                 # оба файла из data/raw/
    python data_prep.py --sample 200000 # быстрый прогон на первых N строках

Что получается в parquet:
    rowid, event_id, user_id, item_id, timestamp (Datetime UTC), time_bucket, day_type,
    history_item_ids (List), history_timestamps (List[Datetime UTC]), hist_len, [clicked]

event_id = строка "user_id_<epoch µs>", если в CSV нет своего event_id.
Детерминированный: одинаковый на любой машине и любой версии polars (в отличие от .hash()).
Время и пиковая память пишутся в data/parquet/prep_report.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

import config as C

ID_HIST = "history_item_ids"
TS_HIST = "history_timestamps"


def peak_rss_gb() -> float | None:
    """Пиковая память процесса в ГБ. Linux: KB, macOS: байты, Windows: через psutil (если стоит)."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024**3 if sys.platform == "darwin" else rss / 1024**2
    except ImportError:  # Windows
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1024**3
        except Exception:
            return None


def sniff(path: Path) -> pl.DataFrame:
    """Читаем небольшой кусок, чтобы понять типы колонок."""
    return pl.read_csv(path, n_rows=5_000, infer_schema_length=5_000)


def epoch_unit(sample_value: float) -> str:
    # секунды ~1.7e9, миллисекунды ~1.7e12, микросекунды ~1.7e15
    if sample_value > 1e14:
        return "us"
    if sample_value > 1e11:
        return "ms"
    return "s"


def ts_expr(col: str, dtype: pl.DataType, sample) -> pl.Expr:
    """Скалярный timestamp -> Datetime(UTC)."""
    if dtype.is_numeric():
        unit = epoch_unit(float(sample))
        return pl.from_epoch(pl.col(col).cast(pl.Int64), time_unit=unit).dt.replace_time_zone("UTC")
    return pl.col(col).str.to_datetime(time_zone="UTC")


def build_lazy(path: Path, n_rows: int | None) -> tuple[pl.LazyFrame, dict]:
    head = sniff(path)
    cols = head.columns
    info = {"columns": cols, "dtypes": {c: str(t) for c, t in head.schema.items()}}

    # Тип item_id определяет тип элементов в истории.
    item_dtype = head.schema["item_id"]
    list_item_dtype = pl.Int64 if item_dtype.is_integer() else pl.String

    # Формат timestamp в истории: смотрим на первую непустую историю.
    ts_dtype = head.schema["timestamp"]
    ts_sample = head["timestamp"].drop_nulls()[0]
    hist_sample = next((h for h in head[TS_HIST].drop_nulls() if h not in ("", "[]")), "[]")
    hist_first = json.loads(hist_sample)[0] if json.loads(hist_sample) else ts_sample
    hist_is_numeric = isinstance(hist_first, (int, float))

    # Всё читаем как строки, кроме того, что уже знаем — меньше сюрпризов при стриминге.
    schema_overrides = {ID_HIST: pl.String, TS_HIST: pl.String}
    lf = pl.scan_csv(path, schema_overrides=schema_overrides, n_rows=n_rows,
                     infer_schema_length=10_000)

    if "rowid" not in cols:
        print(f"  ! в {path.name} нет колонки rowid — создаю row index с 0. "
              "Для evaluation проверьте, что организаторы ждут именно такие rowid!")
        lf = lf.with_row_index("rowid")

    if hist_is_numeric:
        unit = epoch_unit(float(hist_first))
        hist_ts = (pl.col(TS_HIST).fill_null("[]").str.json_decode(pl.List(pl.Int64))
                   .list.eval(pl.from_epoch(pl.element(), time_unit=unit)
                              .dt.replace_time_zone("UTC")))
    else:
        hist_ts = (pl.col(TS_HIST).fill_null("[]").str.json_decode(pl.List(pl.String))
                   .list.eval(pl.element().str.to_datetime(time_zone="UTC")))

    lf = lf.with_columns(
        ts_expr("timestamp", ts_dtype, ts_sample).alias("timestamp"),
        pl.col(ID_HIST).fill_null("[]").str.json_decode(pl.List(list_item_dtype)).alias(ID_HIST),
        hist_ts.alias(TS_HIST),
    ).with_columns(
        pl.col(ID_HIST).list.len().cast(pl.Int16).alias("hist_len"),
    )

    if "event_id" not in cols:
        # Не .hash(): он меняется между версиями polars -> у команды разъехались бы event_id.
        lf = lf.with_columns(pl.concat_str(
            pl.col("user_id").cast(pl.String), pl.col("timestamp").dt.epoch("us").cast(pl.String),
            separator="_").alias("event_id"))

    if "clicked" in cols:
        lf = lf.with_columns(pl.col("clicked").cast(pl.Int8))

    front = ["rowid", "event_id", "user_id", "item_id", "timestamp"]
    lf = lf.select(front + [c for c in lf.collect_schema().names() if c not in front])
    return lf, info


def sanity(pq: Path) -> dict:
    """Быстрые проверки готового parquet (читаем только нужные колонки)."""
    lf = pl.scan_parquet(pq)
    names = lf.collect_schema().names()
    has_label = "clicked" in names

    agg = [pl.len().alias("rows"), pl.col("timestamp").first().alias("ts"),
           pl.col("user_id").first().alias("user_id"), pl.col("hist_len").first().alias("hist_len")]
    if has_label:
        agg.append(pl.col("clicked").sum().alias("clicks"))
    ev = lf.group_by("event_id").agg(agg).collect()

    out = {
        "rows": int(ev["rows"].sum()),
        "events": ev.height,
        "users": ev["user_id"].n_unique(),
        "rows_per_event": {str(k): int(v) for k, v in ev["rows"].value_counts().sort("rows").iter_rows()},
        "ts_min": str(ev["ts"].min()),
        "ts_max": str(ev["ts"].max()),
        "empty_history_share": round(float((ev["hist_len"] == 0).mean()), 4),
        "rowid_unique": lf.select(pl.col("rowid").n_unique()).collect().item() == int(ev["rows"].sum()),
    }
    if has_label:
        out["events_with_click_share"] = round(float((ev["clicks"] > 0).mean()), 4)
        out["events_with_>1_click"] = int((ev["clicks"] > 1).sum())
        out["row_ctr"] = round(float(ev["clicks"].sum() / ev["rows"].sum()), 5)
    return out


def convert(src: Path, dst: Path, n_rows: int | None) -> dict:
    print(f"→ {src.name} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    lf, info = build_lazy(src, n_rows)
    lf.sink_parquet(dst, compression="zstd", row_group_size=500_000)
    secs = time.perf_counter() - t0
    report = {
        "seconds": round(secs, 1),
        "peak_rss_gb_so_far": None if (m := peak_rss_gb()) is None else round(m, 2),
        "csv_gb": round(src.stat().st_size / 1024**3, 3),
        "parquet_gb": round(dst.stat().st_size / 1024**3, 3),
        "source": info,
    }
    report["sanity"] = sanity(dst)
    print(json.dumps({k: v for k, v in report.items() if k != "source"}, indent=2, ensure_ascii=False))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=C.TRAIN_CSV)
    ap.add_argument("--eval", type=Path, default=C.EVAL_CSV)
    ap.add_argument("--sample", type=int, default=None, help="взять только первые N строк каждого CSV")
    args = ap.parse_args()

    report = {
        "train": convert(args.train, C.TRAIN_PQ, args.sample),
        "evaluation": convert(args.eval, C.EVAL_PQ, args.sample),
        "polars": pl.__version__,
    }
    ev = report["evaluation"]["sanity"]
    if args.sample is None and ev["rows"] != C.EXPECTED_EVAL_ROWS:
        print(f"  ! evaluation: {ev['rows']} строк, ожидалось {C.EXPECTED_EVAL_ROWS}")
    for name in ("train", "evaluation"):
        rpe = report[name]["sanity"]["rows_per_event"]
        if args.sample is None and set(rpe) != {str(C.CANDIDATES_PER_EVENT)}:
            print(f"  ! {name}: rows_per_event = {rpe}. Не все события по 10 строк — "
                  "скорее всего у юзера два события в один timestamp склеились. Разобраться ДО сплита.")
    (C.PQ_DIR / "prep_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Готово. Отчёт: {C.PQ_DIR / 'prep_report.json'}")


if __name__ == "__main__":
    main()
