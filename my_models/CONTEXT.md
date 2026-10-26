# CONTEXT for Cursor (read fully before any task)

Hackathon: click prediction, metric = global ROC AUC. I am Participant 4 (Roma): I produce
"user-history vs candidate" SCORES (MF, item2vec, SASRec) that Participant 3 joins by `rowid`
to Nazar's features and feeds into LightGBM. I only edit files inside `my_models/`.
Never edit other files in the repo. Never read or print raw data rows into chat. No git commands.

## Data (already prepared by repo scripts, polars + parquet)
- data/split/fit.parquet   : train events before 25 Jan (config.VAL_START)
- data/split/val.parquet   : first event of each user in 25 Jan..end of train. ONLY for evaluation, never train on it
- data/parquet/train.parquet, data/parquet/evaluation.parquet : full train / final target (no clicked)
- Columns: rowid (opaque TEXT, never convert), event_id (text), user_id (text), item_id (text),
  timestamp (UTC datetime), time_bucket, day_type, history_item_ids (list of text, <=20, oldest->newest,
  may be empty, same item may repeat), history_timestamps (list[datetime]), hist_len, clicked (train only).
- 1 event = 10 candidate rows with identical user, timestamp, history. At most 1 clicked per event;
  ~37% of events have no click (all 10 rows are 0). Row CTR ~6.3%.
- Existing repo helpers: config.py (paths, windows()), metrics.evaluate(preds) where preds has
  columns rowid, predicted_probability (prints global / within-event / event-level AUC), asof.py.
- Facts: ~53k users, evaluation = 1 event per user; ~2.7% eval users not in train, ~1.3% eval events have
  empty history; some history items may not appear in train (warm-up clicks).
- Machine: 15 GB RAM, no GPU, 20 CPU cores. Memory is tight: load only needed columns, work on EVENT level.

## Time windows (STRICT, only via config.windows(final), never hardcode dates)
windows(final=False): part = fit.parquet, w["part_end"]=VAL_START, w["stack_start"]=part_end-2 days.
windows(final=True) : part = train.parquet, target = evaluation.parquet.
For a model in mode M:
  1. Fit model A on part events with timestamp < stack_start.
     Score events in [stack_start, part_end)  -> `<model>_stack.parquet` (booster trains on these).
  2. Fit a fresh model B on ALL part events (< part_end).
     Score target rows (val.parquet, or evaluation.parquet in final) -> `<model>_target.parquet`.
A model must never score rows it was trained on (leak). Vocabulary is built from the fitting part only.

## Architecture (keep simple, readable, explainable to a jury)
my_models/
  common.py   load parquet (needed columns only), build EventTable, Vocab, save_scores, eval helpers
  dataset.py  EventTable -> tensors: history [E,20] (item ids, left-padded 0), hist_mask, cand [E,10], label [E,10], rowid [E,10]
  mf.py       model 1 (later)   item2vec.py, sasrec.py (later)
  run_scores.py  CLI: --model mf --mode val|final --sample 0.1 (fraction of events for quick tests)
  test_scores.py checks
  SCORES.md   documentation for Participant 3 (columns, version, date, standalone AUC, how to rerun)
Every model class exposes: fit(events: EventTable) and score(events: EventTable) -> polars DataFrame[rowid, <name>_score].
Output: data/scores/<mode>/<model>_stack.parquet and <model>_target.parquet, columns rowid + score. mode = val|final.

## Rules / edge cases
- Vocab: item_id -> int. 0 = PAD, 1 = UNK (items not in fitting part). Built from candidates AND histories of fitting part.
- Empty history: must give a finite score (fallback: item popularity from fitting part, or 0). No NaN/inf ever.
- Masked mean for history (exclude PAD). Repeated history items are kept.
- No per-user embedding: user vector comes from history, so new users work.
- Keep 10-candidate structure: arrays shaped [events, 10]; rowid kept alongside, join by rowid as TEXT.
- Fixed seed, all hyperparameters in one dataclass/config block at top of file.
- Assert on output: row count == target rows, rowid unique, all scores finite.
- Type hints, short docstrings, English comments that explain WHY. Small functions. No notebooks.
- Use polars for tables, torch for models, CPU only. Use torch.set_num_threads(16).
- Do not pip install anything without telling me; add new deps to requirements.txt only if I approve.
