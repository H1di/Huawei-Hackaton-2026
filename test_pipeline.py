"""Самопроверка пайплайна на фейковых данных. Запуск: python test_pipeline.py  (или pytest)

Проверяет главное:
  1. prior_stats совпадает с «тупым» перебором (строка видит только события строго раньше себя);
  2. в признак строки не попадает её собственная метка;
  3. fit/val не пересекаются, val = первое событие юзера в окне;
  4. event_id одинаковый при повторной конвертации (детерминированный).
"""
import random

import polars as pl

import config as C
from asof import assert_past_only, prior_stats


def _brute(rows: pl.DataFrame, hist: pl.DataFrame, key: str, rowid) -> tuple[int, int]:
    r = rows.filter(pl.col("rowid") == rowid).row(0, named=True)
    h = hist.filter((pl.col(key) == r[key]) & (pl.col("timestamp") < r["timestamp"]))
    return h.height, int(h["clicked"].sum())


def test_prior_stats_matches_bruteforce():
    fit = pl.read_parquet(C.FIT_PQ)
    for key in ("item_id", "user_id"):
        st = prior_stats(fit, fit, key).join(fit.select("rowid"), on="rowid")
        assert st.height == fit.height, "prior_stats потерял/задублировал строки"
        for rid in random.Random(0).sample(fit["rowid"].to_list(), 300):
            got = st.filter(pl.col("rowid") == rid).row(0, named=True)
            assert (got[f"{key}_n_rows"], got[f"{key}_n_clicks"]) == _brute(fit, fit, key, rid), (key, rid)


def test_own_label_never_leaks():
    """Флипнем метки одного события — его собственные признаки не должны измениться."""
    fit = pl.read_parquet(C.FIT_PQ)
    ev = fit.filter(pl.col("clicked") == 1)["event_id"][0]
    flipped = fit.with_columns(
        pl.when(pl.col("event_id") == ev).then(1 - pl.col("clicked")).otherwise(pl.col("clicked"))
        .cast(pl.Int8).alias("clicked"))
    mine = fit.filter(pl.col("event_id") == ev)
    a = prior_stats(mine, fit, "user_id").sort("rowid")
    b = prior_stats(mine, flipped, "user_id").sort("rowid")
    assert a.drop("user_id_ctr").equals(b.drop("user_id_ctr")), "метка события попала в его же признаки"


def test_val_sees_all_fit():
    fit, val = pl.read_parquet(C.FIT_PQ), pl.read_parquet(C.VAL_PQ)
    assert_past_only(val, fit)
    st = prior_stats(val, fit, "item_id")
    total = fit.group_by("item_id").len()
    chk = val.select("rowid", "item_id").join(st, on="rowid").join(total, on="item_id", how="left")
    assert (chk["item_id_n_rows"] == chk["len"].fill_null(0)).all()


def test_split_shape():
    fit, val = pl.read_parquet(C.FIT_PQ), pl.read_parquet(C.VAL_PQ)
    assert fit["timestamp"].max() < C.VAL_START <= val["timestamp"].min()
    assert val.group_by("user_id").agg(pl.col("event_id").n_unique())["event_id"].max() == 1
    assert (val.group_by("event_id").len()["len"] == C.CANDIDATES_PER_EVENT).all()


def test_event_id_deterministic():
    from data_prep import build_lazy
    a = build_lazy(C.TRAIN_CSV, 5_000)[0].select("event_id").collect()
    b = build_lazy(C.TRAIN_CSV, 5_000)[0].select("event_id").collect()
    assert a.equals(b) and a["event_id"].dtype == pl.String


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("Все проверки прошли")
