"""Бустинг (Участник 3): склеивает признаки Назара и скоры Ромы по rowid, учит LightGBM.

    python booster.py --mode val            # учим на stack-окне fit, AUC на val
    python booster.py --mode final          # учим на stack-окне train, пишем submission.csv
    python booster.py --mode val --scores item2vec,mf   # только выбранные скоры

Откуда что берётся (всё через config.windows(), даты руками не пишем):

  mode   | обучение бустинга                          | предсказание
  -------|--------------------------------------------|--------------------------------
  val    | fit.parquet, строки [stack_start, part_end) | val.parquet
         | + data/features/fit_features.parquet       | + data/features/val_features.parquet
         | + data/scores/val/<m>_stack.parquet        | + data/scores/val/<m>_target.parquet
  final  | train.parquet, те же окна со сдвигом       | evaluation.parquet
         | + data/features/train_features.parquet     | + data/features/eval_features.parquet
         | + data/scores/final/<m>_stack.parquet      | + data/scores/final/<m>_target.parquet

Почему учимся только на stack-окне (последние STACK_DAYS дней части): скоры Ромы
честно посчитаны только там (модель A не видела эти строки). На более ранних строках
скоров нет, а подставить скоры модели B нельзя — она на них училась (утечка).
"""
import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

import config as C

FEAT_DIR = C.ROOT / "data" / "features"
SCORES_DIR = C.ROOT / "data" / "scores"
OUT_DIR = C.ROOT / "data" / "booster"

KEYS = ["rowid", "event_id", "user_id", "timestamp", "clicked"]   # не признаки


@dataclass
class Params:
    """Все гиперпараметры в одном месте. Менять тут, а не в коде."""
    lgb: dict = field(default_factory=lambda: {
        "objective": "binary",        # лидерборд = глобальный AUC по всем строкам -> обычная бинарная задача
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 200,      # данных много, мелкие листья = шум
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 42,
        "num_threads": 0,             # 0 = все ядра
    })
    max_rounds: int = 3000
    early_stop: int = 100
    holdout_share: float = 0.1        # доля ПОЛЬЗОВАТЕЛЕЙ stack-окна для ранней остановки


# ---------------------------------------------------------------- загрузка

def paths(mode: str) -> dict:
    """Все входные файлы для режима. Одна таблица, чтобы val и final не разъехались."""
    if mode == "val":
        return {"train_rows": C.FIT_PQ, "train_feats": FEAT_DIR / "fit_features.parquet",
                "target_rows": C.VAL_PQ, "target_feats": FEAT_DIR / "val_features.parquet",
                "scores": SCORES_DIR / "val", "final": False}
    return {"train_rows": C.TRAIN_PQ, "train_feats": FEAT_DIR / "train_features.parquet",
            "target_rows": C.EVAL_PQ, "target_feats": FEAT_DIR / "eval_features.parquet",
            "scores": SCORES_DIR / "final", "final": True}


def find_scores(score_dir: Path, wanted: list[str] | None) -> list[str]:
    """Имена моделей, у которых есть ОБА файла: <m>_stack и <m>_target."""
    have = sorted(p.name.removesuffix("_stack.parquet") for p in score_dir.glob("*_stack.parquet")
                  if (score_dir / p.name.replace("_stack", "_target")).exists())
    if wanted is None:
        return have
    missing = set(wanted) - set(have)
    if missing:
        raise FileNotFoundError(f"нет скоров {sorted(missing)} в {score_dir} (есть: {have})")
    return wanted


def load_part(rows: pl.LazyFrame, feats: Path, score_dir: Path, models: list[str], kind: str) -> pl.DataFrame:
    """rows (ключи + метка) + признаки Назара + скоры Ромы, всё по rowid.

    Join по признакам inner + проверка числа строк: если у Назара строк меньше,
    падаем, а не теряем строки молча (в val это испортит AUC, в final — сабмит).
    """
    if not feats.exists():
        raise FileNotFoundError(f"нет {feats} — сначала `python features.py` (Назар)")
    n_rows = rows.select(pl.len()).collect().item()

    f = pl.scan_parquet(feats)
    leak = {"clicked", "user_id", "item_id", "timestamp"} & set(f.collect_schema().names())
    if "clicked" in leak:
        raise ValueError(f"в {feats.name} есть clicked — это утечка, признаки должны быть без меток")
    f = f.drop(list(leak))   # сырые id/время как признаки не нужны: они не обобщаются на новое время

    df = rows.join(f, on="rowid", how="inner")
    for m in models:
        s = pl.scan_parquet(score_dir / f"{m}_{kind}.parquet")
        df = df.join(s, on="rowid", how="left")
    df = df.collect()

    if df.height != n_rows:
        raise ValueError(f"{feats.name}: признаки нашлись для {df.height:,} из {n_rows:,} строк")
    for col in [c for c in df.columns if c.endswith("_score")]:
        n_null = df[col].null_count()
        if n_null:
            raise ValueError(f"{col} ({kind}): нет скора у {n_null:,} строк — окна у Ромы и тут не совпадают?")
    return df


def load(mode: str, models_arg: list[str] | None) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    p = paths(mode)
    w = C.windows(final=p["final"])
    models = find_scores(p["scores"], models_arg)
    print(f"режим {mode}: окно бустинга [{w['stack_start']:%d.%m %H:%M}, {w['part_end']:%d.%m %H:%M}), "
          f"скоры: {models or 'НЕТ (только признаки Назара)'}")

    cols = ["rowid", "event_id", "user_id", "timestamp", "clicked"]
    train_rows = (pl.scan_parquet(p["train_rows"]).select(cols)
                  .filter(pl.col("timestamp") >= w["stack_start"], pl.col("timestamp") < w["part_end"]))
    target_rows = pl.scan_parquet(p["target_rows"])
    target_cols = [c for c in cols if c in target_rows.collect_schema().names()]   # в evaluation нет clicked
    target_rows = target_rows.select(target_cols)

    train = load_part(train_rows, p["train_feats"], p["scores"], models, "stack")
    target = load_part(target_rows, p["target_feats"], p["scores"], models, "target")
    return train, target, models


# ---------------------------------------------------------------- признаки бустинга

def add_event_features(df: pl.DataFrame) -> pl.DataFrame:
    """Для каждого скора Ромы: как кандидат выглядит на фоне остальных 9 в своём событии.

    Сырой скор сравнивает кандидата с историей; «лучший ли он в событии» — то, что
    реально решает, по какому из 10 кликнут. Своя метка сюда не попадает: считаем
    только по скорам, которые известны без clicked.
    """
    score_cols = [c for c in df.columns if c.endswith("_score")]
    return df.with_columns(
        *[(pl.col(c) - pl.col(c).max().over("event_id")).alias(f"{c}_vs_event_max") for c in score_cols],
        *[pl.col(c).rank("average", descending=True).over("event_id").alias(f"{c}_rank_in_event")
          for c in score_cols],
    )


def encode_strings(train: pl.DataFrame, target: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Текстовые колонки (например time_bucket='00-06') -> номер. Словарь только из train,
    чтобы одно и то же значение получило один и тот же номер в обеих таблицах."""
    for c in [c for c, t in train.schema.items() if t == pl.String and c not in KEYS]:
        vocab = {v: i for i, v in enumerate(sorted(train[c].drop_nulls().unique().to_list()))}
        enc = pl.col(c).replace_strict(vocab, default=-1, return_dtype=pl.Int32)
        train, target = train.with_columns(enc), target.with_columns(enc)
    return train, target


def feature_columns(train: pl.DataFrame, target: pl.DataFrame) -> list[str]:
    feats = [c for c in train.columns if c not in KEYS]
    missing = set(feats) - set(target.columns)
    if missing:   # разные версии признаков у train и target = модель увидит мусор
        raise ValueError(f"в target нет колонок {sorted(missing)} — пересчитайте признаки одной версией")
    return feats


# ---------------------------------------------------------------- обучение

def to_xy(df: pl.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    x = df.select(pl.col(feats).cast(pl.Float32)).to_numpy()   # null -> nan, LightGBM их понимает
    y = df["clicked"].to_numpy() if "clicked" in df.columns else None
    return x, y


def fit(train: pl.DataFrame, feats: list[str], prm: Params) -> tuple[lgb.Booster, int]:
    """1) Ранняя остановка: 10% пользователей stack-окна откладываем и ищем число деревьев.
       2) Переобучаем на ВСЁМ окне с найденным числом деревьев (+10%, т.к. данных стало больше).
    Делим по пользователям, а не по строкам: иначе 10 строк одного события и соседние события
    того же юзера окажутся по обе стороны, и остановка будет слишком оптимистичной."""
    hold = (train["user_id"].hash(seed=0) % 1000) < int(prm.holdout_share * 1000)
    tr, ho = train.filter(~hold), train.filter(hold)
    xtr, ytr = to_xy(tr, feats)
    xho, yho = to_xy(ho, feats)

    booster = lgb.train(prm.lgb, lgb.Dataset(xtr, ytr, feature_name=feats),
                        num_boost_round=prm.max_rounds,
                        valid_sets=[lgb.Dataset(xho, yho, feature_name=feats)], valid_names=["holdout"],
                        callbacks=[lgb.early_stopping(prm.early_stop, verbose=False), lgb.log_evaluation(200)])
    best = booster.best_iteration or prm.max_rounds
    print(f"ранняя остановка: {best} деревьев, holdout AUC {booster.best_score['holdout']['auc']:.4f}")

    n_full = int(best * 1.1)
    x, y = to_xy(train, feats)
    full = lgb.train(prm.lgb, lgb.Dataset(x, y, feature_name=feats), num_boost_round=n_full)
    return full, n_full


def print_importance(booster: lgb.Booster, top: int = 25) -> None:
    """Gain-важность: какие признаки реально двигают модель. Нулевые — кандидаты на удаление."""
    gain = booster.feature_importance("gain")
    order = np.argsort(-gain)
    total = gain.sum() or 1.0
    print("\nважность признаков (доля gain):")
    for i in order[:top]:
        print(f"  {booster.feature_name()[i]:<40} {gain[i] / total:6.1%}")
    zero = [booster.feature_name()[i] for i in order if gain[i] == 0]
    if zero:
        print(f"  не используются: {zero}")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "final"], required=True)
    ap.add_argument("--scores", default=None, help="через запятую, напр. item2vec,mf; по умолчанию все найденные")
    ap.add_argument("--out", default="submission.csv", help="куда писать сабмит (только final)")
    args = ap.parse_args()
    prm = Params()

    train, target, models = load(args.mode, args.scores.split(",") if args.scores else None)
    train, target = add_event_features(train), add_event_features(target)
    train, target = encode_strings(train, target)
    feats = feature_columns(train, target)
    print(f"обучение: {train.height:,} строк, {train['clicked'].mean():.2%} кликов; "
          f"предсказание: {target.height:,} строк; признаков {len(feats)}")

    booster, n_trees = fit(train, feats, prm)
    print_importance(booster)

    pred = target.select("rowid", pl.Series("predicted_probability", booster.predict(to_xy(target, feats)[0])))
    assert pred["rowid"].n_unique() == pred.height, "дубли rowid"
    assert pred["predicted_probability"].is_finite().all(), "NaN/inf в предсказаниях"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(OUT_DIR / f"lgb_{args.mode}.txt"))
    pred.write_parquet(OUT_DIR / f"pred_{args.mode}.parquet")   # для блендинга/сравнения версий
    report = {"mode": args.mode, "scores": models, "n_features": len(feats), "n_trees": n_trees,
              "train_rows": train.height, "params": asdict(prm)}

    if args.mode == "val":
        from metrics import evaluate
        report["val_auc"] = evaluate(pred)
        # Сравнение: насколько бустинг лучше лучшего одиночного скора Ромы
        for m in models:
            col = f"{m}_score"
            print(f"  для сравнения, один {col}: AUC {roc_auc_score(target['clicked'], target[col]):.4f}")
    else:
        from check_submission import check
        pred.write_csv(args.out)
        errors = check(Path(args.out))
        if errors:
            raise SystemExit("САБМИТ СЛОМАН:\n  - " + "\n  - ".join(errors))
        print(f"{args.out}: OK, можно отправлять")

    (OUT_DIR / f"report_{args.mode}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
