"""Маленький синтетический датасет в формате соревнования — для проверки пайплайна без 7 ГБ."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

rng = np.random.default_rng(1)
N_USERS, N_ITEMS = 3000, 500
T0 = int(datetime(2025, 1, 20, tzinfo=timezone.utc).timestamp())
T_TRAIN_END = int(datetime(2025, 1, 28, tzinfo=timezone.utc).timestamp())
BUCKETS = ["00-06", "06-12", "12-18", "18-24"]
iso = len(sys.argv) > 2 and sys.argv[2] == "iso"   # проверить и ISO-формат времени


def fmt(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if iso else t


def events_for(users, t_lo, t_hi, label: bool, one_per_user=False):
    rows, rowid = [], 0
    hist = {u: [(int(i), T0 - int(rng.integers(1, 86400))) for i in rng.integers(0, N_ITEMS, rng.integers(0, 4))]
            for u in users}
    for u in users:
        n_ev = 1 if one_per_user else int(rng.integers(1, 6))
        for t in sorted(rng.integers(t_lo, t_hi, n_ev)):
            cands = rng.choice(N_ITEMS, 10, replace=False)
            h = hist[u][-20:]
            p_click = 0.6 if len(h) else 0.3
            clicked_idx = int(rng.integers(0, 10)) if rng.random() < p_click else -1
            for k, item in enumerate(cands):
                hour = datetime.fromtimestamp(int(t), timezone.utc).hour
                r = {"user_id": int(u), "item_id": int(item), "timestamp": fmt(int(t)),
                     "time_bucket": BUCKETS[hour // 6],
                     "day_type": "weekend" if datetime.fromtimestamp(int(t), timezone.utc).weekday() >= 5 else "weekday",
                     "history_item_ids": json.dumps([i for i, _ in h]),
                     "history_timestamps": json.dumps([fmt(x) for _, x in h])}
                if label:
                    r["clicked"] = int(k == clicked_idx)
                rows.append(r)
            if clicked_idx >= 0:
                hist[u].append((int(cands[clicked_idx]), int(t)))
    return rows


out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
train_users = np.arange(N_USERS)
pl.DataFrame(events_for(train_users, T0, T_TRAIN_END, True)).write_csv(out / "train.csv")
eval_users = np.concatenate([rng.choice(train_users, 800, replace=False), np.arange(N_USERS, N_USERS + 200)])
ev = pl.DataFrame(events_for(eval_users, T_TRAIN_END, T_TRAIN_END + 86400, False, one_per_user=True))
ev.with_row_index("rowid", offset=1_000_000).write_csv(out / "evaluation.csv")
print("fake data ->", out)
