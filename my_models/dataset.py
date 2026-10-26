"""EventTable + Vocab -> CPU tensors. History is left-padded so index -1 is newest."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from common import MAX_HIST, N_CAND, EventTable, Vocab

torch.set_num_threads(16)


@dataclass
class EventTensors:
    history: torch.Tensor  # [E, 20] long, PAD=0 on the left
    hist_mask: torch.Tensor  # [E, 20] bool, False on PAD
    cand: torch.Tensor  # [E, 10] long
    label: torch.Tensor | None  # [E, 10] float, or None
    rowid: np.ndarray  # [E, 10] object/str, never cast to int


def to_tensors(events: EventTable, vocab: Vocab) -> EventTensors:
    """Encode items with the fitting-part vocab. Unknown -> UNK=1."""
    n = len(events)
    history = np.zeros((n, MAX_HIST), dtype=np.int64)
    hist_mask = np.zeros((n, MAX_HIST), dtype=bool)
    cand = np.zeros((n, N_CAND), dtype=np.int64)

    hist_lists = events.history.to_list()
    cand_lists = events.item_ids.to_list()
    rowids = np.array(events.rowids.to_list(), dtype=object)

    for i, h in enumerate(hist_lists):
        seq = vocab.encode_seq(h)[-MAX_HIST:]
        k = len(seq)
        if k:
            history[i, MAX_HIST - k :] = seq
            hist_mask[i, MAX_HIST - k :] = True
        cand[i] = vocab.encode_seq(cand_lists[i])

    label = None
    if events.labels is not None:
        label = torch.tensor(events.labels.to_list(), dtype=torch.float32)

    return EventTensors(
        history=torch.from_numpy(history),
        hist_mask=torch.from_numpy(hist_mask),
        cand=torch.from_numpy(cand),
        label=label,
        rowid=rowids,
    )
