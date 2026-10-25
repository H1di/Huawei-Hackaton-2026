# Huawei hackathon — данные и валидация (Участник 1, Влад)

## Запуск

```bash
pip install -r requirements.txt
# положить train.csv и evaluation.csv в data/raw/
python data_prep.py --sample 200000   # 1) быстрый прогон: проверить, что формат распознаётся
python data_prep.py                   # 2) полный прогон (пишет время и память в data/parquet/prep_report.json)
python split.py                       # 3) fit/val сплит + сравнение val с evaluation
python metrics.py                     # 4) самопроверка: случайные скоры ≈ 0.5
python test_pipeline.py               # 5) тесты: нет утечек, сплит правильный
```

Проверить всё без реальных данных: `python make_fake_data.py data/raw` (или `... data/raw iso`), потом шаги 2–5.
Проверено на polars 1.44 и 2.0, Linux/macOS/Windows.

`--sample` режет по строкам, последнее событие может обрезаться — только для проверки формата.

## Что получается

| Файл | Что внутри |
|---|---|
| `data/parquet/train.parquet` | весь train, истории уже списками |
| `data/parquet/evaluation.parquet` | evaluation, тот же формат без `clicked` |
| `data/split/fit.parquet` | события до 25.01 — источник статистик и обучающих строк (см. правила ниже) |
| `data/split/val.parquet` | первое событие каждого юзера 25–27.01 — **только проверка** |

Колонки: `rowid, event_id, user_id, item_id, timestamp (UTC), time_bucket, day_type, history_item_ids (list), history_timestamps (list[datetime]), hist_len, clicked`.

Если в CSV нет `event_id`, он = строка `"<user_id>_<timestamp в µs>"` — одинаковая на любой машине и версии polars.
Проверка: в отчёте `rows_per_event` должно быть только `10`. Если есть `20` — у юзера два события в одну секунду склеились, `data_prep.py` об этом предупредит.

## Для всей команды

```python
from metrics import evaluate
auc = evaluate(preds)   # preds: rowid, predicted_probability по val.parquet
```

Печатает три числа:
- **global AUC** — то, что считает лидерборд;
- **within-event** — правильно ли ранжируем 10 кандидатов (растёт от признаков «кандидат vs история»);
- **event-level** — отличаем ли события с кликом от событий без клика (растёт от признаков активности юзера).

Перед КАЖДОЙ отправкой:
```bash
python check_submission.py submission.csv   # exit 1 = не отправлять
```

## Правила против утечек

Главная ловушка: посчитать CTR юзера/товара по всему fit и на этом же fit учить модель. Тогда в признак строки входит её же `clicked` — модель переобучается на CTR, а на val/evaluation он слабее. Как считать винрейт героя, включая матч, исход которого предсказываешь.

**1. Счётчики и CTR (Назар) — только через `asof.prior_stats`.** Строка видит лишь события строго раньше своего `timestamp`, одна функция на все случаи:

```python
from asof import prior_stats
prior_stats(fit, fit, "item_id")              # обучение: строка fit видит только более ранний fit
prior_stats(val, fit, "item_id")              # валидация: val видит весь fit
prior_stats(evaluation, train, "item_id")     # финал
prior_stats(fit, fit, ["user_id", "item_id"]) # составной ключ
```
Отдаёт `rowid` + `n_rows, n_clicks, n_events, n_click_events, ctr` (сглаженный), `secs_since_prev`.

**2. Скоры моделей (Рома: MF, item2vec, ко-клики, SASRec)** построчно as-of не посчитать, поэтому окна из `config.windows()`:
- модель учится на событиях `< stack_start` и скорит строки `[stack_start, part_end)` — на них учится бустинг (Евгений);
- для предсказания val/evaluation модель переобучается на всей части и скорит целевые строки;
- `windows()` для валидации, `windows(final=True)` для финала — сдвиг одинаковый.
- `asof.assert_past_only(rows, history)` — упадёт, если history пересекается с rows или заходит в их время.

**3. Признаки из истории** (`history_item_ids/timestamps`, время с последнего клика, был ли кандидат в истории) утечек не дают — история и так только до события.

**4. Финал:** те же скрипты, часть = весь `train.parquet`, целевые строки = `evaluation.parquet`.

**5.** Не тюним на публичный лидерборд — решает приватный.

## Проверить на реальных данных

- Есть ли в `evaluation.csv` колонка `rowid`. Если нет, скрипт создаст индекс с 0 и предупредит. Тогда уточнить у организаторов формат.
- Формат `timestamp` (unix или ISO) определяется сам, проверить `ts_min/ts_max` в отчёте.
- В `split_report.json` сравнить `new_users_share` и `empty_history_share` у val и evaluation. Если сильно расходятся, двигаем `VAL_START` в `config.py`.
