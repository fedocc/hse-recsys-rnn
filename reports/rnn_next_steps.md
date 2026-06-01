# Следующие шаги по RNN-ветке

RNN-ветка как отдельная исследовательская часть закрыта. Новые крупные эксперименты сейчас не нужны: основной кандидат выбран и проверен на нескольких seed.

## Зафиксировано

- общее разбиение пользователей из `master_split_lesha.csv`;
- честный prefix-based протокол удержания;
- baseline для next-event prediction против `GRU/LSTM`: `MostPopular` и `Markov-1`;
- ablation способа pooling скрытых состояний;
- supervised fine-tuning;
- negative controls;
- ablation длины префикса на общей валидной когорте;
- order sensitivity analysis;
- перебор ёмкости модели;
- финальная проверка устойчивости к случайной инициализации.

Финальный RNN-кандидат:

```text
GRU, hidden_dim=256, num_layers=1, max pooling, prefix_len=150
```

Стабильный результат на задаче удержания с `Baseline + embedding`:

| target | ROC-AUC mean ± std | PR-AUC mean ± std |
|---|---:|---:|
| retention_7d | 0.6922 ± 0.0052 | 0.4978 ± 0.0023 |
| retention_14d | 0.7051 ± 0.0032 | 0.3644 ± 0.0037 |

## Что делать дальше

1. Синхронизировать с командой общий data contract:
   - `appmetrica_device_id`;
   - `split`;
   - `retention_7d`;
   - `retention_14d`;
   - `prefix_len`;
   - `emb_000...`.

2. Прогнать RNN, SimCLR и BERT4Rec embeddings через один общий скрипт оценки:
   - `Baseline`;
   - `Embedding only`;
   - `Baseline + Embedding`;
   - одинаковые XGBoost seeds;
   - одинаковые `ROC-AUC`, `PR-AUC`, `num_users`, `positive_rate`.

3. Собрать финальный narrative:
   - RNN хорошо учит структуру последовательности событий;
   - способ pooling и capacity важны;
   - рост качества next-event prediction не гарантирует лучший retention;
   - финальный RNN-кандидат даёт устойчивый, но умеренный прирост на задаче удержания;
   - следующее честное сравнение должно быть только в общем evaluator.

Сделано: ключевые RNN-серии вынесены в `scripts/run_rnn_experiments.py`. Раннер не переобучает тяжелые модели с нуля, но по `configs/rnn_experiments.yaml` валидирует и материализует таблицы, графики и manifest для финальных артефактов.
