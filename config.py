"""Общие пути и константы. Меняйте здесь, а не в скриптах."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"          # сюда кладём train.csv / evaluation.csv
PQ_DIR = ROOT / "data" / "parquet"       # train.parquet / evaluation.parquet
SPLIT_DIR = ROOT / "data" / "split"      # fit.parquet / val.parquet

TRAIN_CSV = RAW_DIR / "train.csv"
EVAL_CSV = RAW_DIR / "evaluation.csv"
TRAIN_PQ = PQ_DIR / "train.parquet"
EVAL_PQ = PQ_DIR / "evaluation.parquet"
FIT_PQ = SPLIT_DIR / "fit.parquet"
VAL_PQ = SPLIT_DIR / "val.parquet"

# Всё до VAL_START -> fit, начиная с VAL_START -> валидационное окно.
VAL_START = datetime(2025, 1, 25, tzinfo=timezone.utc)
TRAIN_END = datetime(2025, 1, 28, tzinfo=timezone.utc)   # train = 20–27 янв включительно

# Скоры моделей (MF, item2vec, ко-клики, SASRec) нельзя посчитать «as-of» для каждой строки.
# Поэтому: такая модель учится на событиях < stack_start, бустинг учится на строках
# [stack_start, part_end). Для предсказания модель переобучается на всей части.
# Сдвиг одинаковый для валидации и финала -> схемы не разъезжаются.
STACK_DAYS = 2


def windows(final: bool = False) -> dict:
    """final=False: валидация (часть = fit, предсказываем val).
    final=True:  финал (часть = весь train, предсказываем evaluation)."""
    end = TRAIN_END if final else VAL_START
    return {"part_end": end, "stack_start": end - timedelta(days=STACK_DAYS)}


EXPECTED_EVAL_ROWS = 529_390
EXPECTED_EVAL_EVENTS = 52_939
CANDIDATES_PER_EVENT = 10
