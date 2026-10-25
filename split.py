"""Шаг 2. Валидационная схема, повторяющая настоящий evaluation.

    fit  = все события train с timestamp <  VAL_START (25.01)
    val  = ПЕРВОЕ событие каждого пользователя с timestamp >= VAL_START
    остальные события окна выкидываются (в настоящем evaluation их тоже нет)

Правило против утечек (подробно — README и asof.py):
  * счётчики/CTR — только через asof.prior_stats: строка видит лишь события строго раньше себя;
  * скоры моделей (MF, SASRec, ко-клики) — модель учится на fit < stack_start,
    бустинг учится на строках fit >= stack_start (config.windows());
  * val/evaluation видят всю свою обучающую часть целиком.
Финальный сабмит: та же схема, но часть = весь train, а val = evaluation (config.windows(final=True)).

Запуск: python split.py
"""
import json

import polars as pl

import config as C


def event_table(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.group_by("event_id").agg(
        pl.col("user_id").first(),
        pl.col("timestamp").first(),
        pl.col("hist_len").first(),
        *([pl.col("clicked").max().alias("has_click")]
          if "clicked" in lf.collect_schema().names() else []),
    )


def profile(events: pl.DataFrame, known_users: pl.Series, name: str) -> dict:
    """Тот же набор метрик для val и evaluation — чтобы сравнить, похожи ли они."""
    out = {
        "events": events.height,
        "users": events["user_id"].n_unique(),
        "new_users_share": round(float((~events["user_id"].is_in(known_users.implode())).mean()), 4),
        "empty_history_share": round(float((events["hist_len"] == 0).mean()), 4),
        "hist_len_mean": round(float(events["hist_len"].mean()), 2),
        "ts_min": str(events["timestamp"].min()),
        "ts_max": str(events["timestamp"].max()),
    }
    if "has_click" in events.columns:
        out["events_with_click_share"] = round(float(events["has_click"].mean()), 4)
    print(f"[{name}] " + json.dumps(out, ensure_ascii=False))
    return out


def main():
    C.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train = pl.scan_parquet(C.TRAIN_PQ)

    ev = event_table(train).collect()
    fit_ev = ev.filter(pl.col("timestamp") < C.VAL_START)
    win_ev = ev.filter(pl.col("timestamp") >= C.VAL_START)
    val_ev = win_ev.sort("timestamp", "event_id").unique("user_id", keep="first", maintain_order=True)

    assert fit_ev.height and val_ev.height, "fit или val пустой — проверьте VAL_START и даты в данных"
    assert fit_ev["timestamp"].max() < val_ev["timestamp"].min(), "утечка по времени"

    train.join(fit_ev.lazy().select("event_id"), on="event_id", how="semi") \
         .sink_parquet(C.FIT_PQ, compression="zstd")
    train.join(val_ev.lazy().select("event_id"), on="event_id", how="semi") \
         .sink_parquet(C.VAL_PQ, compression="zstd")

    # Сравнение val с настоящим evaluation: «новые» юзеры = не встречались в своей обучающей части.
    report = {
        "val_start": str(C.VAL_START),
        "fit": {"events": fit_ev.height, "rows": pl.scan_parquet(C.FIT_PQ).select(pl.len()).collect().item()},
        "dropped_window_events": win_ev.height - val_ev.height,
        "val": profile(val_ev, fit_ev["user_id"].unique(), "val"),
    }
    if C.EVAL_PQ.exists():
        eval_ev = event_table(pl.scan_parquet(C.EVAL_PQ)).collect()
        report["evaluation"] = profile(eval_ev, ev["user_id"].unique(), "evaluation")
    print(f"[fit] {report['fit']}, выкинуто событий окна: {report['dropped_window_events']}")

    w = C.windows()
    report["stack_window"] = {"stack_start": str(w["stack_start"]),
                              "booster_events": fit_ev.filter(pl.col("timestamp") >= w["stack_start"]).height}
    (C.SPLIT_DIR / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Готово: {C.FIT_PQ}, {C.VAL_PQ}")


if __name__ == "__main__":
    main()
