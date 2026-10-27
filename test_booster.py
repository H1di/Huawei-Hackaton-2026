"""Самопроверка бустинга без реальных данных. Запуск: python test_booster.py  (или pytest)

Проверяет:
  1. признаки «кандидат на фоне события» считаются внутри своего события и не трогают метку;
  2. текстовые колонки кодируются одинаково в train и target;
  3. LightGBM находит заложенный сигнал и выдаёт по прогнозу на каждую строку;
  4. tl2vec: история выравнивается вправо, пустая история / незнакомый товар дают 0, а не NaN.
"""
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from booster import Params, add_event_features, encode_strings, feature_columns, fit, to_xy


def _toy(n_events: int, seed: int) -> pl.DataFrame:
    """События по 10 кандидатов; клик чаще у кандидата с большим m_score."""
    rng = np.random.default_rng(seed)
    n = n_events * 10
    score = rng.normal(size=n)
    clicked = (score + rng.normal(size=n) > 2.0).astype(np.int8)
    return pl.DataFrame({
        "rowid": [f"r{seed}_{i}" for i in range(n)],
        "event_id": np.repeat(np.arange(n_events), 10).astype(str),
        "user_id": np.repeat(np.arange(n_events) % 500, 10).astype(str),
        "timestamp": np.zeros(n, dtype=np.int64),
        "clicked": clicked,
        "m_score": score,
        "time_bucket": rng.choice(["00-06", "06-12", "12-18", "18-24"], n),
    })


def test_event_features_stay_inside_event():
    df = pl.DataFrame({"event_id": ["a"] * 3 + ["b"] * 3, "m_score": [1.0, 3.0, 2.0, 10.0, 20.0, 30.0],
                       "clicked": [1, 0, 0, 0, 0, 1]})
    out = add_event_features(df)
    assert out["m_score_vs_event_max"].to_list() == [-2.0, 0.0, -1.0, -20.0, -10.0, 0.0]
    assert out["m_score_rank_in_event"].to_list() == [3.0, 1.0, 2.0, 3.0, 2.0, 1.0]
    assert out.drop("m_score_vs_event_max", "m_score_rank_in_event").equals(df)   # метку не трогали


def test_string_encoding_consistent():
    tr = pl.DataFrame({"rowid": ["1", "2"], "time_bucket": ["06-12", "00-06"]})
    te = pl.DataFrame({"rowid": ["3", "4"], "time_bucket": ["00-06", "новое"]})
    tr2, te2 = encode_strings(tr, te)
    assert tr2["time_bucket"].to_list() == [1, 0]
    assert te2["time_bucket"].to_list() == [0, -1]   # невиданное значение -> -1, а не падение


def test_booster_learns_signal():
    train, target = encode_strings(*map(add_event_features, (_toy(3000, 1), _toy(1000, 2))))
    feats = feature_columns(train, target)
    prm = Params(max_rounds=200, early_stop=20)
    booster, _ = fit(train, feats, prm)
    x, y = to_xy(target, feats)
    p = booster.predict(x)
    assert len(p) == target.height and np.isfinite(p).all()
    assert roc_auc_score(y, p) > 0.85, roc_auc_score(y, p)


def test_tl2vec_history_and_edge_cases():
    from gensim.models import Word2Vec
    from tl2vec import Cfg, Embeddings, encode, features, hist_matrix

    e = Embeddings(Word2Vec([["a", "b", "c"], ["b", "c", "d"]] * 20, vector_size=4, window=5,
                            min_count=1, seed=0, workers=1))
    ia, ib, ic = (e.lookup.filter(pl.col("item_id") == x)["i"].item() for x in "abc")
    H = hist_matrix(pl.Series([["a", "b"], [], ["zzz", "c"]]), e)
    assert H[0].tolist() == [e.unk] * 18 + [ia, ib]      # свежий клик в последней колонке
    assert (H[1] == e.unk).all()                          # пустая история
    assert H[2].tolist() == [e.unk] * 19 + [ic]           # незнакомый товар истории = пустое место

    f = features(e, H, np.array([0, 1, 2]), encode(pl.Series(["c", "c", "zzz"]), e), Cfg(chunk_rows=2))
    assert all(np.isfinite(v).all() for v in f.values())
    assert f["score"][1] == f["max"][1] == f["decay"][1] == 0   # пустая история -> 0
    assert f["score"][2] == f["vnorm"][2] == 0                   # незнакомый кандидат -> 0
    assert abs(f["max"][0] - max(e.unit[ic] @ e.unit[ia], e.unit[ic] @ e.unit[ib])) < 1e-5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("Все проверки прошли")
