"""Статистики «на момент события» — без target leakage. Через это считаем ВСЕ счётчики/CTR.

Проблема, которую решает:
    Если CTR юзера/товара посчитать по всему fit и на том же fit учить модель,
    в признак строки попадает её же метка clicked. Модель переобучается на CTR,
    а на val/evaluation этот признак уже честный и слабее -> модель хуже, чем могла быть.

Правило:
    признак строки = агрегат только по событиям history, у которых timestamp СТРОГО меньше,
    чем у этой строки. Своё событие (и всё в ту же секунду) не попадает никогда.

Одна функция на все три случая — схема одинаковая, поэтому train и predict не разъезжаются:

    from asof import prior_stats
    fit = pl.read_parquet(C.FIT_PQ); val = pl.read_parquet(C.VAL_PQ)

    # обучение: каждая строка fit видит только более ранний fit
    tr_item = prior_stats(fit, fit, "item_id")
    # валидация: val видит весь fit (fit целиком раньше val)
    va_item = prior_stats(val, fit, "item_id")
    # финал: train = весь train.parquet, eval видит весь train
    ev_item = prior_stats(evaluation, train, "item_id")

Ключ может быть составным: prior_stats(fit, fit, ["user_id", "item_id"]).
На выходе: rowid + колонки с префиксом (по умолчанию <ключ>_):
    n_rows    — сколько раз показывали (для товара = показы)
    n_clicks  — сколько кликов
    n_events  — в скольких событиях встречался (для юзера = число сессий)
    n_click_events — событий с кликом (для юзера: «активность»)
    ctr       — сглаженный CTR: (n_clicks + alpha * global_ctr) / (n_rows + alpha)
    secs_since_prev — секунд с прошлого появления ключа (null, если не было)
"""
from collections.abc import Sequence

import polars as pl


def prior_stats(rows: pl.DataFrame | pl.LazyFrame,
                history: pl.DataFrame | pl.LazyFrame,
                key: str | Sequence[str],
                alpha: float = 20.0,
                prefix: str | None = None) -> pl.DataFrame:
    keys = [key] if isinstance(key, str) else list(key)
    prefix = prefix if prefix is not None else "_".join(keys) + "_"
    rows, history = rows.lazy(), history.lazy()

    if "clicked" not in history.collect_schema().names():
        raise ValueError("history без clicked — статистики можно считать только по размеченным данным")

    global_ctr = history.select(pl.col("clicked").mean()).collect().item()

    # 1) сворачиваем history до (ключ, timestamp): всё в одну секунду — одна точка
    per_ts = (
        history.group_by(keys + ["timestamp"])
        .agg(pl.len().alias("r"),
             pl.col("clicked").sum().alias("c"),
             pl.col("event_id").n_unique().alias("e"),
             pl.col("event_id").filter(pl.col("clicked") == 1).n_unique().alias("ce"))
        .sort(keys + ["timestamp"])
        # 2) накопленные суммы ВКЛЮЧАЯ эту точку
        .with_columns(pl.col(c).cum_sum().over(keys).alias(f"cum_{c}") for c in ("r", "c", "e", "ce"))
        .select(keys + [pl.col("timestamp").alias("prev_ts"),
                        pl.col("timestamp").alias("_ts"),
                        "cum_r", "cum_c", "cum_e", "cum_ce"])
    )

    # 3) для каждой строки берём последнюю точку СТРОГО раньше её timestamp
    q = rows.select(["rowid", "timestamp"] + keys).sort("timestamp")
    joined = q.join_asof(per_ts.sort("_ts"), left_on="timestamp", right_on="_ts",
                         by=keys, strategy="backward", allow_exact_matches=False,
                         check_sortedness=False)  # обе стороны отсортированы по времени выше

    n_rows = pl.col("cum_r").fill_null(0)
    n_clicks = pl.col("cum_c").fill_null(0)
    return joined.select(
        "rowid",
        n_rows.cast(pl.Int32).alias(f"{prefix}n_rows"),
        n_clicks.cast(pl.Int32).alias(f"{prefix}n_clicks"),
        pl.col("cum_e").fill_null(0).cast(pl.Int32).alias(f"{prefix}n_events"),
        pl.col("cum_ce").fill_null(0).cast(pl.Int32).alias(f"{prefix}n_click_events"),
        ((n_clicks + alpha * global_ctr) / (n_rows + alpha)).cast(pl.Float32).alias(f"{prefix}ctr"),
        (pl.col("timestamp") - pl.col("prev_ts")).dt.total_seconds().cast(pl.Float32)
        .alias(f"{prefix}secs_since_prev"),
    ).collect()


def assert_past_only(rows: pl.DataFrame, history: pl.DataFrame) -> None:
    """Для признаков, посчитанных НЕ через prior_stats (эмбеддинги, MF, ко-клики целиком по history):
    такие можно применять только к rows, которые целиком позже history (val от fit, eval от train).
    На строки самого fit их вешать нельзя — там своя метка внутри. Падает, если правило нарушено."""
    shared = rows.select("event_id").unique().join(history.select("event_id").unique(), on="event_id")
    if shared.height:
        raise AssertionError(f"{shared.height} событий есть и в rows, и в history — утечка метки")
    if history["timestamp"].max() >= rows["timestamp"].min():
        raise AssertionError("history заходит во время rows — утечка по времени")
