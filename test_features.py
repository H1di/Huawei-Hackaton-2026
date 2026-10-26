"""Проверки для features.py. Запуск: python test_features.py  (или pytest test_features.py)

Что проверяем:
  1. форма: строк столько же, сколько во входе, rowid уникальны и совпадают;
  2. диапазоны: CTR в [0, 1], счётчики >= 0 и т.д.;
  3. статистики товара/пользователя/пары совпадают с «тупым» перебором;
  4. признаки «кандидат в истории» совпадают с ручным подсчётом;
  5. утечки нет: если перевернуть метки события, его признаки не меняются;
  6. признаки внутри события (ранг, отклонение от среднего) устроены правильно;
  7. val похож на fit по распределениям (только предупреждения).

Для скорости большинство проверок идут на выборке из ~40 событий.
Требуются готовые fit.parquet / val.parquet (python data_prep.py && python split.py).
"""
import random
from functools import lru_cache

import polars as pl

import config as C
from features import build

SAMPLE_EVENTS = 40      # сколько событий берём для проверок «вручную»
SAMPLE_ROWS = 30        # сколько строк сверяем перебором
HIST_COLS = ["rowid", "event_id", "user_id", "item_id", "timestamp", "clicked"]


# ---------- общие данные (читаем один раз) ----------

@lru_cache(maxsize=1)
def fit_hist() -> pl.DataFrame:
    """fit в роли «истории» для prior_stats: нужны только эти колонки."""
    return pl.read_parquet(C.FIT_PQ, columns=HIST_COLS)


@lru_cache(maxsize=1)
def sample_rows() -> pl.DataFrame:
    """Все колонки для ~40 случайных событий из fit."""
    events = fit_hist()["event_id"].unique().sort().to_list()
    pick = random.Random(0).sample(events, min(SAMPLE_EVENTS, len(events)))
    return (pl.scan_parquet(C.FIT_PQ)
            .filter(pl.col("event_id").is_in(pick))
            .collect())


@lru_cache(maxsize=1)
def sample_features() -> pl.DataFrame:
    return build(sample_rows(), fit_hist())


@lru_cache(maxsize=1)
def val_rows() -> pl.DataFrame:
    return pl.read_parquet(C.VAL_PQ)


@lru_cache(maxsize=1)
def val_features() -> pl.DataFrame:
    return build(val_rows(), fit_hist())


def _pick_rowids(df: pl.DataFrame, n: int) -> list:
    return random.Random(1).sample(df["rowid"].to_list(), min(n, df.height))


# ---------- 1. форма ----------

def test_shape_and_keys():
    val, f = val_rows(), val_features()
    assert f.height == val.height, f"строк {f.height}, а во входе {val.height} (join размножил/потерял строки)"
    assert f["rowid"].n_unique() == f.height, "rowid в признаках повторяются"
    assert set(f["rowid"].to_list()) == set(val["rowid"].to_list()), "наборы rowid не совпали"
    assert "event_id" not in f.columns and "clicked" not in f.columns, "в признаках не должно быть event_id/clicked"


# ---------- 2. диапазоны ----------

def test_value_ranges():
    f = val_features()
    for c in [c for c in f.columns if c.endswith("_ctr")]:
        lo, hi = f[c].min(), f[c].max()
        assert 0 <= lo and hi <= 1, f"{c} вне [0, 1]: {lo}..{hi}"
    for c in [c for c in f.columns if c.endswith(("_n_rows", "_n_clicks", "_n_events", "_n_click_events"))]:
        assert f[c].min() >= 0, f"{c} < 0"
    assert f["cand_in_hist_cnt"].null_count() == 0, "cand_in_hist_cnt должен быть 0, а не null"
    assert 0 <= f["cand_in_hist_cnt"].min() and f["cand_in_hist_cnt"].max() <= 20
    assert f["cand_pos_from_end"].drop_nulls().is_between(0, 19).all(), "cand_pos_from_end вне 0..19"
    assert f["hour"].is_between(0, 23).all()
    assert f["user_no_click_share"].is_between(0, 1).all()
    # null в cand_* допустим только когда кандидата нет в истории
    no_cand = f.filter(pl.col("cand_in_hist_cnt") == 0)
    assert no_cand["cand_pos_from_end"].null_count() == no_cand.height
    assert no_cand["cand_secs_since_click"].null_count() == no_cand.height


def test_candidate_found_in_history_at_all():
    """Если везде 0 — скорее всего разные типы item_id и истории (int против str)."""
    f = val_features()
    assert f["cand_in_hist_cnt"].max() > 0, \
        "ни один кандидат не найден в истории: проверьте dtype item_id и history_item_ids"


# ---------- 3. статистики против перебора ----------

def test_prior_stats_match_bruteforce():
    rows, f, hist = sample_rows(), sample_features(), fit_hist()
    keys = {"item_id": ["item_id"], "user_id": ["user_id"], "user_id_item_id": ["user_id", "item_id"]}
    for rid in _pick_rowids(rows, SAMPLE_ROWS):
        r = rows.filter(pl.col("rowid") == rid).row(0, named=True)
        got = f.filter(pl.col("rowid") == rid).row(0, named=True)
        for prefix, cols in keys.items():
            cond = pl.col("timestamp") < r["timestamp"]
            for c in cols:
                cond = cond & (pl.col(c) == r[c])
            past = hist.filter(cond)
            assert got[f"{prefix}_n_rows"] == past.height, (prefix, rid, "n_rows")
            assert got[f"{prefix}_n_clicks"] == int(past["clicked"].sum() or 0), (prefix, rid, "n_clicks")


# ---------- 4. кандидат и история против ручного подсчёта ----------

def test_candidate_history_features_bruteforce():
    rows, f = sample_rows(), sample_features()
    by_rowid = {r["rowid"]: r for r in f.iter_rows(named=True)}
    for r in rows.iter_rows(named=True):
        ids, tss = r["history_item_ids"], r["history_timestamps"]
        matches = [i for i, x in enumerate(ids) if x == r["item_id"]]
        got = by_rowid[r["rowid"]]

        assert got["cand_in_hist_cnt"] == len(matches), (r["rowid"], "cand_in_hist_cnt")
        if matches:
            assert got["cand_pos_from_end"] == len(ids) - 1 - matches[-1], (r["rowid"], "cand_pos_from_end")
            expect = min((r["timestamp"] - tss[i]).total_seconds() for i in matches)
            assert got["cand_secs_since_click"] == expect, (r["rowid"], "cand_secs_since_click")
        else:
            assert got["cand_pos_from_end"] is None and got["cand_secs_since_click"] is None, r["rowid"]


def test_secs_since_last_click():
    rows, f = sample_rows(), sample_features()
    by_rowid = {r["rowid"]: r for r in f.iter_rows(named=True)}
    for r in rows.iter_rows(named=True):
        got = by_rowid[r["rowid"]]["secs_since_last_click"]
        if r["history_timestamps"]:
            assert got == (r["timestamp"] - r["history_timestamps"][-1]).total_seconds(), r["rowid"]
        else:
            assert got is None, f"пустая история, а secs_since_last_click = {got}"
        assert by_rowid[r["rowid"]]["hist_len"] == len(r["history_item_ids"])


# ---------- 5. утечки ----------

def test_own_label_never_leaks():
    """Перевернём метки одного события: его собственные признаки не должны измениться.
    CTR чуть-чуть сдвигается (глобальный CTR зависит от всех меток) — для них допуск 1e-3."""
    hist, rows = fit_hist(), sample_rows()
    ev = rows.filter(pl.col("clicked") == 1)["event_id"][0]
    mine = rows.filter(pl.col("event_id") == ev)
    flipped = hist.with_columns(
        pl.when(pl.col("event_id") == ev).then(1 - pl.col("clicked")).otherwise(pl.col("clicked"))
        .cast(pl.Int8).alias("clicked"))

    a = build(mine, hist).sort("rowid")
    b = build(mine, flipped).sort("rowid")

    approx = [c for c in a.columns if c.endswith("ctr") or c == "ctr_vs_event_mean"]
    skip = approx + ["ctr_rank_in_event"]           # ранг может чуть сдвинуться из-за сглаживания
    exact = [c for c in a.columns if c not in skip]
    assert a.select(exact).equals(b.select(exact)), "метка события попала в его же признаки"
    for c in approx:
        diff = (a[c].cast(pl.Float64) - b[c].cast(pl.Float64)).abs().max()
        assert diff is None or diff < 1e-3, f"{c} заметно изменился при смене метки: {diff}"


def test_val_features_use_only_fit():
    """val не должен пересекаться с fit по событиям и времени."""
    fit_ev, val = fit_hist(), val_rows()
    assert fit_ev["timestamp"].max() < val["timestamp"].min(), "fit заходит во время val"
    shared = set(fit_ev["event_id"].unique().to_list()) & set(val["event_id"].unique().to_list())
    assert not shared, f"{len(shared)} событий есть и в fit, и в val"


# ---------- 6. признаки внутри события ----------

def test_within_event_features():
    rows, f = sample_rows(), sample_features()
    j = f.join(rows.select("rowid", "event_id"), on="rowid")
    g = j.group_by("event_id").agg(
        pl.len().alias("n"),
        pl.col("ctr_rank_in_event").sort().alias("ranks"),
        pl.col("ctr_vs_event_mean").sum().alias("dev_sum"),
    )
    assert (g["n"] == C.CANDIDATES_PER_EVENT).all(), "в событии не 10 строк"
    expected = list(range(1, C.CANDIDATES_PER_EVENT + 1))
    assert all(r.to_list() == expected for r in g["ranks"]), "ранги в событии должны быть 1..10 без повторов"
    assert g["dev_sum"].abs().max() < 1e-4, "отклонения от среднего по событию должны суммироваться в 0"


# ---------- 7. fit против val (только предупреждения) ----------

def test_val_distribution_close_to_fit():
    f_s, v = sample_features(), val_features()
    for c in ["hist_len", "item_id_ctr", "user_id_ctr", "cand_in_hist_cnt", "user_no_click_share"]:
        a, b = f_s[c].mean(), v[c].mean()
        if a and abs(a - b) / abs(a) > 0.5:
            print(f"  ! предупреждение: среднее {c} сильно отличается (выборка fit {a:.4f}, val {b:.4f})")
    print("  (сравнение распределений: смотрите предупреждения выше; выборка fit маленькая, это ориентир)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("Все проверки прошли")