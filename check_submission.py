"""Проверка сабмита перед отправкой. ЗАПУСКАТЬ ПЕРЕД КАЖДОЙ ОТПРАВКОЙ (лимит 10 в день).

    python check_submission.py submission.csv

Проверяет: заголовок ровно `rowid,predicted_probability`, 529 390 строк,
нет дублей, набор rowid совпадает с evaluation (если есть evaluation.parquet),
нет пустых/NaN/inf, все значения в [0, 1]. Код выхода 1 — если хоть что-то не так.
"""
import sys
from pathlib import Path

import polars as pl

import config as C

HEADER = "rowid,predicted_probability"


def check(path: Path) -> list[str]:
    errors: list[str] = []
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline().strip()
    if header != HEADER:
        errors.append(f"заголовок '{header}', нужен '{HEADER}'")
        return errors

    df = pl.read_csv(path, schema_overrides={"predicted_probability": pl.String})
    if df.height != C.EXPECTED_EVAL_ROWS:
        errors.append(f"строк {df.height:,}, нужно {C.EXPECTED_EVAL_ROWS:,}")

    if df["rowid"].null_count():
        errors.append(f"пустых rowid: {df['rowid'].null_count()}")
    dups = df.filter(pl.col("rowid").is_duplicated())["rowid"].n_unique()
    if dups:
        errors.append(f"дублирующихся rowid: {dups}")

    p = df["predicted_probability"].cast(pl.Float64, strict=False)
    bad_parse = (p.is_null() & df["predicted_probability"].is_not_null()).sum()
    if bad_parse:
        errors.append(f"не числа в predicted_probability: {bad_parse}")
    if p.null_count() - bad_parse:
        errors.append(f"пустых значений: {p.null_count() - bad_parse}")
    finite = p.drop_nulls()
    n_nan_inf = int((finite.is_nan() | finite.is_infinite()).sum())
    if n_nan_inf:
        errors.append(f"NaN/inf: {n_nan_inf}")
    out_of_range = int(((finite < 0) | (finite > 1)).sum())
    if out_of_range:
        errors.append(f"вне [0, 1]: {out_of_range} (min {finite.min()}, max {finite.max()})")

    if C.EVAL_PQ.exists():
        ref = pl.read_parquet(C.EVAL_PQ, columns=["rowid"])["rowid"]
        sub = df["rowid"].cast(ref.dtype, strict=False)
        missing = (~ref.is_in(sub.implode())).sum()
        extra = (~sub.is_in(ref.implode())).sum()
        if missing:
            errors.append(f"нет {missing} rowid из evaluation")
        if extra:
            errors.append(f"{extra} лишних rowid, которых нет в evaluation")
    else:
        print(f"  (нет {C.EVAL_PQ} — совпадение набора rowid не проверено)")

    if not errors and finite.n_unique() == 1:
        print("  ! все предсказания одинаковые — AUC будет 0.5")
    return errors


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python check_submission.py submission.csv")
    errors = check(Path(sys.argv[1]))
    if errors:
        print("СЛОМАНО:\n  - " + "\n  - ".join(errors))
        sys.exit(1)
    print("OK, можно отправлять")


if __name__ == "__main__":
    main()
