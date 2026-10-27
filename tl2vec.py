"""Скоры «вкус юзера vs кандидат» на word2vec по ХРОНОЛОГИЯМ кликов (Участник 3).

    python tl2vec.py --mode val      # data/scores/val/tl2vec_{stack,target}.parquet
    python tl2vec.py --mode final    # data/scores/final/tl2vec_{stack,target}.parquet

booster.py подхватывает их сам (как и item2vec Ромы). Окна те же, что у Ромы (config.windows()):
  модель A учится на событиях < stack_start и скорит stack-окно (на нём учится бустинг);
  модель B учится на всей части и скорит target (val или evaluation).

Чем отличается от my_models/item2vec.py (замеры на val, один скор, global AUC):
  * одно «предложение» на юзера — все его клики по времени, без повторов. У Ромы предложение =
    история каждого события, и одни и те же 20 кликов юзера идут десятки раз со сдвигом;
  * 16 измерений вместо 32: 8 -> 0.703, 12 -> 0.759, 16 -> 0.765, 24 -> 0.758, 64 -> 0.734.
    Похоже, у генератора данных скрытое пространство небольшой размерности, лишние оси = шум;
  * 3 сида, скоры усредняются: 0.764 -> 0.769.

Почему только такие колонки: модели A и B — два разных обучения, их пространства повёрнуты
случайно относительно друг друга. Бустинг учится на скорах A, а предсказывает по скорам B,
поэтому годятся только величины, которые от поворота не зависят (косинусы, длины векторов).
Покоординатные признаки (u_k * v_k) из A в B не переносятся.

Колонки (всё усреднено по сидам):
  tl2vec_score        косинус кандидата со средним (единичных) векторов истории — главный скор
  tl2vec_inout_score  средний входной вектор истории · выходной вектор кандидата
                      (ровно то, что word2vec учится предсказывать: «какой товар рядом»)
  tl2vec_max_score    максимальный косинус кандидата с одним товаром истории
  tl2vec_decay_score  косинус с историей, свежие клики весят больше (decay^возраст)
  tl2vec_vnorm_score  длина вектора кандидата: сколько данных о товаре видела модель
  tl2vec_unorm_score  длина среднего единичного вектора истории: 1 = узкий вкус, 0 = разброс
Пустая история или незнакомый товар дают 0, а не NaN.
"""
import argparse
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from gensim.models import Word2Vec

import config as C

NAME = "tl2vec"
MAX_HIST = 20
SCORES_DIR = C.ROOT / "data" / "scores"


@dataclass
class Cfg:
    dim: int = 16
    window: int = 20             # вся история: важен вкус, а не порядок (окно 5 -> AUC 0.53)
    epochs: int = 10             # 30 эпох не лучше
    negative: int = 10
    seeds: tuple = (0, 1, 2)
    decay: float = 0.8           # вес клика = decay ** (сколько кликов было после него)
    workers: int = 10
    chunk_rows: int = 200_000    # строк за раз при скоринге: [chunk, 20, dim] float32 ~ 250 MB


# ---------------------------------------------------------------- обучение

def timelines(part: pl.LazyFrame) -> list[list[str]]:
    """Клики каждого юзера по времени: истории его событий + клики в самих событиях, без повторов.
    Только размеченная часть: в историях — клики строго до события, в clicked — свои события."""
    one = part.unique("event_id")                     # у 10 строк события история одна и та же
    hist = (one.select("user_id", "history_item_ids", "history_timestamps")
            .explode("history_item_ids", "history_timestamps").drop_nulls()
            .select("user_id", pl.col("history_item_ids").alias("item_id"),
                    pl.col("history_timestamps").alias("ts")))
    own = part.filter(pl.col("clicked") == 1).select("user_id", "item_id", pl.col("timestamp").alias("ts"))
    tl = (pl.concat([hist, own]).unique().sort("user_id", "ts")
          .group_by("user_id", maintain_order=True).agg("item_id").collect())
    return tl["item_id"].to_list()


class Embeddings:
    """Векторы одного word2vec. Последняя строка — нули: «нет товара» (паддинг или незнакомый)."""

    def __init__(self, m: Word2Vec):
        pad = np.zeros((1, m.wv.vector_size), np.float32)
        self.lookup = pl.DataFrame({"item_id": m.wv.index_to_key,
                                    "i": np.arange(len(m.wv.index_to_key), dtype=np.int32)})
        self.unk = len(m.wv.index_to_key)
        self.win = np.vstack([m.wv.vectors, pad]).astype(np.float32)     # входные: товар как контекст
        self.wout = np.vstack([m.syn1neg, pad]).astype(np.float32)       # выходные: товар как «ответ»
        self.unit = self.win / np.linalg.norm(self.win, axis=1, keepdims=True).clip(1e-9)


def fit(part: pl.LazyFrame, cfg: Cfg) -> list[Embeddings]:
    t = time.time()
    sents = timelines(part)
    out = [Embeddings(Word2Vec(sents, vector_size=cfg.dim, window=cfg.window, min_count=1, sg=1,
                               negative=cfg.negative, epochs=cfg.epochs, workers=cfg.workers, seed=s))
           for s in cfg.seeds]
    print(f"  {len(sents):,} юзеров, {sum(map(len, sents)):,} кликов, "
          f"{len(cfg.seeds)} модели за {time.time() - t:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------- скоринг

def encode(items: pl.Series, e: Embeddings) -> np.ndarray:
    """item_id -> номер строки в векторах (незнакомый -> e.unk)."""
    return (items.to_frame("item_id").join(e.lookup, on="item_id", how="left", maintain_order="left")
            ["i"].fill_null(e.unk).to_numpy())


def hist_matrix(hist: pl.Series, e: Embeddings) -> np.ndarray:
    """Истории событий -> [E, 20], выравнивание вправо: последняя колонка = самый свежий клик."""
    lens = hist.list.len().to_numpy().astype(np.int64)   # uint64 + int64 в numpy даёт float
    assert lens.max(initial=0) <= MAX_HIST, "история длиннее 20"
    flat = encode(hist.explode().drop_nulls(), e)        # пустой список даёт null -> выкинут
    assert len(flat) == lens.sum(), "null внутри истории"
    starts = np.repeat(np.cumsum(lens) - lens, lens)
    H = np.full((len(hist), MAX_HIST), e.unk, np.int32)
    H[np.repeat(np.arange(len(hist)), lens), MAX_HIST - np.repeat(lens, lens) + np.arange(len(flat)) - starts] = flat
    return H


def features(e: Embeddings, H: np.ndarray, ev_idx: np.ndarray, cand: np.ndarray, cfg: Cfg) -> dict:
    """Все колонки для строк (кандидат cand[r] в событии ev_idx[r]) по одной модели."""
    mask = H != e.unk                                                           # [E,20]
    nh = mask.sum(1)
    U = (e.unit[H] * mask[..., None]).sum(1) / nh.clip(1)[:, None]              # средний единичный вектор
    unorm = np.linalg.norm(U, axis=1)
    Uin = (e.win[H] * mask[..., None]).sum(1) / nh.clip(1)[:, None]
    w = mask * cfg.decay ** np.arange(MAX_HIST - 1, -1, -1, dtype=np.float32)   # свежий клик = вес 1

    out = {k: np.empty(len(cand), np.float32) for k in ("score", "inout", "max", "decay", "vnorm")}
    for s in range(0, len(cand), cfg.chunk_rows):
        ei, ci = ev_idx[s:s + cfg.chunk_rows], cand[s:s + cfg.chunk_rows]
        v = e.unit[ci]
        sims = np.einsum("rhd,rd->rh", e.unit[H[ei]], v)                        # [R,20]
        sl = slice(s, s + len(ci))
        out["score"][sl] = np.einsum("rd,rd->r", U[ei], v) / unorm[ei].clip(1e-9)
        out["inout"][sl] = np.einsum("rd,rd->r", Uin[ei], e.wout[ci])
        out["max"][sl] = np.where(nh[ei] > 0, np.where(mask[ei], sims, -np.inf).max(1), 0.0)
        out["decay"][sl] = (sims * w[ei]).sum(1) / w[ei].sum(1).clip(1e-9)
        out["vnorm"][sl] = np.linalg.norm(e.win[ci], axis=1)
    out["unorm"] = unorm[ev_idx].astype(np.float32)
    return out


def score(rows: pl.LazyFrame, models: list[Embeddings], cfg: Cfg) -> pl.DataFrame:
    """rowid + колонки tl2vec_*_score, усреднённые по сидам."""
    ev = rows.select("event_id", "history_item_ids").unique("event_id").collect()
    r = (rows.select("rowid", "event_id", "item_id").collect()
         .join(ev.select("event_id", pl.int_range(pl.len(), dtype=pl.Int32).alias("ev_idx")),
               on="event_id", how="left", maintain_order="left"))
    ev_idx = r["ev_idx"].to_numpy()
    total: dict[str, np.ndarray] = {}
    for e in models:
        f = features(e, hist_matrix(ev["history_item_ids"], e), ev_idx, encode(r["item_id"], e), cfg)
        total = {k: total.get(k, 0) + v / len(models) for k, v in f.items()}
    cols = {(f"{NAME}_score" if k == "score" else f"{NAME}_{k}_score"): v for k, v in total.items()}
    return r.select("rowid").with_columns(**{k: pl.Series(v) for k, v in cols.items()})


def save(df: pl.DataFrame, n_rows: int, mode: str, kind: str) -> None:
    """Те же проверки, что save_scores у Ромы, но колонок несколько (save_scores пишет одну)."""
    assert df.height == n_rows, f"{kind}: строк {df.height}, нужно {n_rows}"
    assert df["rowid"].n_unique() == df.height, f"{kind}: дубли rowid"
    for c in df.columns[1:]:
        assert df[c].is_finite().all(), f"{kind}: NaN/inf в {c}"
    path = SCORES_DIR / mode / f"{NAME}_{kind}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"  записано {path} ({df.height:,} строк)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "final"], required=True)
    mode = ap.parse_args().mode
    final, cfg = mode == "final", Cfg()
    part_pq, target_pq = (C.TRAIN_PQ, C.EVAL_PQ) if final else (C.FIT_PQ, C.VAL_PQ)
    w = C.windows(final=final)
    cols = ["rowid", "event_id", "user_id", "item_id", "timestamp", "history_item_ids",
            "history_timestamps", "clicked"]
    part = pl.scan_parquet(part_pq).select(cols)

    print(f"модель A: события < {w['stack_start']:%d.%m}, скорим stack-окно", flush=True)
    stack = part.filter(pl.col("timestamp") >= w["stack_start"], pl.col("timestamp") < w["part_end"])
    models = fit(part.filter(pl.col("timestamp") < w["stack_start"]), cfg)
    save(score(stack, models, cfg), stack.select(pl.len()).collect().item(), mode, "stack")

    print(f"модель B: вся часть (< {w['part_end']:%d.%m}), скорим {target_pq.name}", flush=True)
    target = pl.scan_parquet(target_pq)
    models = fit(part.filter(pl.col("timestamp") < w["part_end"]), cfg)
    save(score(target, models, cfg), target.select(pl.len()).collect().item(), mode, "target")


if __name__ == "__main__":
    main()
