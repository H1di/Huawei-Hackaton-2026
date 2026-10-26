import polars as pl
import config as C
from asof import prior_stats

def cand_hist_features(rows: pl.DataFrame) -> pl.DataFrame:
    """Для каждой строки: сколько раз кандидат был в истории, где и как давно."""
    ex = (
        rows.lazy()
        .select("rowid", "item_id", "timestamp", "history_item_ids", "history_timestamps")
        .with_columns(pl.int_ranges(pl.col("history_item_ids").list.len()).alias("pos"))
        .explode("history_item_ids", "history_timestamps", "pos")
        .filter(pl.col("history_item_ids") == pl.col("item_id"))
        .group_by("rowid")
        .agg(
            pl.len().cast(pl.Int16).alias("cand_in_hist_cnt"),
            pl.col("pos").max().alias("cand_last_pos"),
            (pl.col("timestamp") - pl.col("history_timestamps"))
                .dt.total_seconds().min().alias("cand_secs_since_click"),
        )
        .collect()
    )
    return ex

def build(rows: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    item = prior_stats(rows, history, "item_id")
    user = prior_stats(rows, history, "user_id")
    pair = prior_stats(rows, history, ["user_id", "item_id"])
    cand = cand_hist_features(rows)

    hist = rows.select(
        "rowid", "event_id", "hist_len", "time_bucket", "day_type",
        pl.col("timestamp").dt.hour().alias("hour"),
        (pl.col("timestamp") - pl.col("history_timestamps").list.last())
            .dt.total_seconds().alias("secs_since_last_click"),
    )

    df = (hist.join(item, on="rowid").join(user, on="rowid").join(pair, on="rowid")
              .join(cand, on="rowid", how="left"))     # left: кандидата может не быть в истории

    df = df.with_columns(
        pl.col("cand_in_hist_cnt").fill_null(0),       # не было в истории -> 0 раз
        # позиция с конца: 0 = самый свежий клик, 19 = самый старый
        (pl.col("hist_len") - 1 - pl.col("cand_last_pos")).alias("cand_pos_from_end"),
        (pl.col("item_id_ctr") - pl.col("item_id_ctr").mean().over("event_id")).alias("ctr_vs_event_mean"),
        pl.col("item_id_ctr").rank("ordinal", descending=True).over("event_id").alias("ctr_rank_in_event"),
        (1 - pl.col("user_id_n_click_events") / pl.col("user_id_n_events").clip(lower_bound=1))
            .alias("user_no_click_share"),
    ).drop("cand_last_pos")
    return df.drop("event_id")

if __name__ == "__main__":
    fit, val = pl.read_parquet(C.FIT_PQ), pl.read_parquet(C.VAL_PQ)
    train, ev = pl.read_parquet(C.TRAIN_PQ), pl.read_parquet(C.EVAL_PQ)
    out = C.ROOT / "data" / "features"; out.mkdir(parents=True, exist_ok=True)

    build(fit, fit).write_parquet(out / "fit_features.parquet")   # для обучения
    build(val, fit).write_parquet(out / "val_features.parquet")   # для проверки
    build(ev, train).write_parquet(out / "eval_features.parquet") # для финала
    