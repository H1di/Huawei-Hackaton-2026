"""evaluate(preds) — глобальный ROC AUC на локальной валидации. Пользуются все.

    from metrics import evaluate
    auc = evaluate(preds)   # preds: DataFrame (polars или pandas) с rowid, predicted_probability

Кроме главного числа печатает разложение, чтобы понимать, ЧТО улучшать:
  * global_auc        — то, что считает лидерборд (все строки разом)
  * within_event_auc  — насколько правильно ранжируем 10 кандидатов внутри события
                        (только события с кликом)
  * event_auc         — отличаем ли события «с кликом» от «без клика»
                        (по максимальному скору в событии)
Растёт within_event — работают признаки «кандидат vs история».
Растёт event_auc   — работают признаки активности пользователя.
"""
from pathlib import Path

import polars as pl
from sklearn.metrics import roc_auc_score

import config as C

_LABELS_CACHE: dict[Path, pl.DataFrame] = {}


def _labels(path: Path) -> pl.DataFrame:
    if path not in _LABELS_CACHE:
        _LABELS_CACHE[path] = pl.read_parquet(path, columns=["rowid", "event_id", "clicked"])
    return _LABELS_CACHE[path]


def _to_polars(preds) -> pl.DataFrame:
    if isinstance(preds, pl.DataFrame):
        df = preds
    else:  # pandas
        df = pl.from_pandas(preds.reset_index(drop=True) if hasattr(preds, "reset_index") else preds)
    missing = {"rowid", "predicted_probability"} - set(df.columns)
    if missing:
        raise ValueError(f"в preds нет колонок: {missing}")
    return df.select(pl.col("rowid"), pl.col("predicted_probability").cast(pl.Float64))


def evaluate(preds, val_path: Path = C.VAL_PQ, verbose: bool = True) -> float:
    lab = _labels(val_path)
    p = _to_polars(preds)

    if p["rowid"].is_duplicated().any():
        raise ValueError("дубли rowid в preds")
    if p["predicted_probability"].is_null().any() or p["predicted_probability"].is_nan().any():
        raise ValueError("NaN/null в predicted_probability")

    df = lab.join(p, on="rowid", how="left")
    n_missing = df["predicted_probability"].null_count()
    if n_missing:
        raise ValueError(f"нет предсказаний для {n_missing} строк валидации")
    extra = p.height - (df.height - n_missing)
    if extra:
        raise ValueError(f"{extra} rowid в preds не из валидации — проверьте, что предсказываете val.parquet")

    y, s = df["clicked"].to_numpy(), df["predicted_probability"].to_numpy()
    auc = float(roc_auc_score(y, s))

    if verbose:
        parts = decompose(df)
        print(f"global AUC {auc:.5f} | within-event {parts['within_event_auc']:.5f} "
              f"| event-level {parts['event_auc']:.5f} | rows {df.height:,}")
    return auc


def decompose(df: pl.DataFrame) -> dict:
    """df: rowid, event_id, clicked, predicted_probability."""
    s = pl.col("predicted_probability")
    # Внутри события: доля негативов ниже кликнутого (ничьи = 0.5), усредняем по событиям с кликом.
    pos = df.filter(pl.col("clicked") == 1).select("event_id", s.alias("p_pos"))
    within = (
        df.filter(pl.col("clicked") == 0)
        .join(pos, on="event_id", how="inner")
        .group_by("event_id")
        .agg(((s < pl.col("p_pos")).cast(pl.Float64)
              + 0.5 * (s == pl.col("p_pos")).cast(pl.Float64)).mean().alias("auc"))
    )
    ev = df.group_by("event_id").agg(pl.col("clicked").max().alias("y"), s.max().alias("m"))
    event_auc = (float(roc_auc_score(ev["y"].to_numpy(), ev["m"].to_numpy()))
                 if ev["y"].n_unique() == 2 else float("nan"))
    return {"within_event_auc": float(within["auc"].mean()), "event_auc": event_auc}


if __name__ == "__main__":
    # Самопроверка: случайные скоры должны давать ~0.5.
    import numpy as np
    lab = _labels(C.VAL_PQ)
    rnd = lab.select("rowid").with_columns(
        pl.Series("predicted_probability", np.random.default_rng(0).random(lab.height)))
    evaluate(rnd)
