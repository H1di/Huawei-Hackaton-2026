"""Item2vec: items that appear in the same click histories get similar vectors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from gensim.models import Word2Vec

from common import EventTable


@dataclass
class I2VConfig:
    DIM: int = 32
    WINDOW: int = 20      # whole history: order matters little, co-occurrence does
    EPOCHS: int = 10
    NEGATIVE: int = 5
    SEED: int = 0


class Item2VecModel:
    name = "item2vec"

    def __init__(self, cfg: I2VConfig | None = None):
        self.cfg = cfg or I2VConfig()
        self.index: dict[str, int] = {}
        self.vecs: np.ndarray = np.zeros((0, self.cfg.DIM), dtype=np.float32)

    def fit(self, events: EventTable) -> None:
        df = events.to_frame()
        labels = df["labels"].to_list() if "labels" in df.columns else None
        sentences: list[list[str]] = []
        for k, (hist, cands) in enumerate(zip(df["history"].to_list(), df["item_ids"].to_list())):
            s = list(hist)
            # The clicked candidate of a fitting event is also a click: add it.
            if labels is not None and 1 in labels[k]:
                s.append(cands[labels[k].index(1)])
            if len(s) >= 2:
                sentences.append(s)
        w2v = Word2Vec(
            sentences, vector_size=self.cfg.DIM, window=self.cfg.WINDOW, min_count=1,
            sg=1, negative=self.cfg.NEGATIVE, epochs=self.cfg.EPOCHS,
            workers=16, seed=self.cfg.SEED,
        )
        words = w2v.wv.index_to_key
        self.index = {w: i for i, w in enumerate(words)}
        v = w2v.wv.vectors.astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        self.vecs = v / np.clip(norms, 1e-9, None)   # unit length: dot = cosine
        print(f"[item2vec] sentences {len(sentences)} items {len(words)}")

    def score(self, events: EventTable) -> pl.DataFrame:
        df = events.to_frame()
        out = np.zeros((len(df), 10), dtype=np.float64)
        D = self.cfg.DIM
        for k, (hist, cands) in enumerate(zip(df["history"].to_list(), df["item_ids"].to_list())):
            h_idx = [self.index[x] for x in hist if x in self.index]
            if not h_idx:
                continue                       # empty/unknown history -> score 0, finite
            user = self.vecs[h_idx].mean(axis=0)
            for j, c in enumerate(cands):
                ci = self.index.get(c)
                if ci is not None:
                    out[k, j] = float(self.vecs[ci] @ user)
        assert np.isfinite(out).all()
        return pl.DataFrame({
            "rowid": df["rowids"].explode().to_list(),
            f"{self.name}_score": out.reshape(-1),
        })
