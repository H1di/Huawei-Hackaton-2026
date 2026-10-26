"""Matrix factorization: history mean vs candidate, two item roles."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import PAD_ID, UNK_ID, EventTable, Vocab
from dataset import EventTensors, to_tensors

torch.set_num_threads(16)

EpochCallback = Callable[[int, "MFModel", float], None]


@dataclass
class MFConfig:
    DIM: int = 32
    LR: float = 5e-3
    WD: float = 1e-6
    BATCH_EVENTS: int = 512
    EPOCHS: int = 10
    SEED: int = 0


class MFNet(nn.Module):
    """hist_emb = item as evidence, cand_emb = item as candidate; PAD/UNK stay 0."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.hist_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.cand_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.item_bias = nn.Embedding(vocab_size, 1, padding_idx=PAD_ID)
        self.global_bias = nn.Parameter(torch.full((1,), -2.7))  # start at the real ~6% click rate
        for emb in (self.hist_emb, self.cand_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=0.05)      # small start, no random noise
        nn.init.zeros_(self.item_bias.weight)
        self.zero_special()

    def zero_special(self) -> None:
        """PAD is frozen by padding_idx; UNK is never trained — keep both at 0."""
        with torch.no_grad():
            for emb in (self.hist_emb, self.cand_emb, self.item_bias):
                emb.weight[PAD_ID].zero_()
                emb.weight[UNK_ID].zero_()

    def forward(
        self,
        history: torch.Tensor,
        hist_mask: torch.Tensor,
        cand: torch.Tensor,
    ) -> torch.Tensor:
        hist = self.hist_emb(history)  # [B, L, D]
        mask = hist_mask.to(dtype=hist.dtype).unsqueeze(-1)
        total = (hist * mask).sum(dim=1)
        # Empty history: total=0 and count clamped to 1 -> user_vec=0, no NaN.
        count = mask.sum(dim=1).clamp(min=1.0)
        user_vec = total / count  # [B, D]
        cand_vec = self.cand_emb(cand)  # [B, 10, D]
        dots = (user_vec.unsqueeze(1) * cand_vec).sum(dim=-1)
        bias = self.item_bias(cand).squeeze(-1)
        return dots + bias + self.global_bias


class MFModel:
    name = "mf"

    def __init__(self, cfg: MFConfig | None = None):
        self.cfg = cfg or MFConfig()
        self.vocab: Vocab | None = None
        self.net: MFNet | None = None

    def fit(
        self,
        events: EventTable,
        *,
        on_epoch: EpochCallback | None = None,
    ) -> None:
        if events.labels is None:
            raise ValueError("MF.fit needs labels")
        cfg = self.cfg
        torch.manual_seed(cfg.SEED)
        self.vocab = Vocab.fit(events)
        self.net = MFNet(len(self.vocab), cfg.DIM)
        print(f"[{self.name}] vocab {len(self.vocab)} events {len(events)}")
        data = to_tensors(events, self.vocab)
        if data.label is None:
            raise ValueError("MF.fit needs labels")
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
        n = len(events)
        bs = cfg.BATCH_EVENTS
        for epoch in range(cfg.EPOCHS):
            self.net.train()
            t0 = time.perf_counter()
            perm = torch.randperm(n)
            total_loss = 0.0
            n_rows = 0
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                logits = self.net(data.history[idx], data.hist_mask[idx], data.cand[idx])
                loss = F.binary_cross_entropy_with_logits(logits, data.label[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                self.net.zero_special()
                total_loss += float(loss.item()) * idx.numel()
                n_rows += int(idx.numel())
            mean_loss = total_loss / max(n_rows, 1)
            dt = time.perf_counter() - t0
            print(f"epoch {epoch + 1}/{cfg.EPOCHS} train_loss {mean_loss:.5f} time {dt:.1f}s")
            if on_epoch is not None:
                on_epoch(epoch, self, mean_loss)
        del data
        self.net.eval()
        self.net.zero_special()

    def score(self, events: EventTable, tensors: EventTensors | None = None) -> pl.DataFrame:
        if self.net is None or self.vocab is None:
            raise RuntimeError("MFModel.score called before fit")
        data = tensors if tensors is not None else to_tensors(events, self.vocab)
        self.net.eval()
        self.net.zero_special()
        bs = self.cfg.BATCH_EVENTS
        n = data.cand.shape[0]
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, bs):
                sl = slice(start, start + bs)
                logits = self.net(data.history[sl], data.hist_mask[sl], data.cand[sl])
                if not torch.isfinite(logits).all():
                    raise ValueError("non-finite mf logits")
                parts.append(logits.cpu().numpy().reshape(-1))
        scores = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        rowids = data.rowid.reshape(-1)
        return pl.DataFrame({
            "rowid": rowids.tolist(),
            f"{self.name}_score": scores.astype(np.float64, copy=False),
        })
