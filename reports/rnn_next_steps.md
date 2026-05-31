# Следующие шаги по RNN-ветке

RNN-ветка как отдельная исследовательская часть закрыта. Новые крупные эксперименты сейчас не нужны: основной кандидат выбран и проверен на нескольких seed.

## Зафиксировано

- общий user-level split из `master_split_lesha.csv`;
- честный prefix-based retention protocol;
- next-event baseline для `GRU/LSTM` против `MostPopular` и `Markov-1`;
- pooling ablation;
- supervised fine-tuning;
- negative controls;
- common-valid prefix ablation;
- order sensitivity analysis;
- capacity sweep;
- final seed stability.

Финальный RNN-кандидат:

```text
GRU, hidden_dim=256, num_layers=1, pooling=max, prefix_len=150
```

Стабильный downstream результат с `Baseline + embedding`:

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

2. Прогнать RNN, SimCLR и BERT4Rec embeddings через один общий downstream evaluator:
   - `Baseline`;
   - `Embedding only`;
   - `Baseline + Embedding`;
   - одинаковые XGBoost seeds;
   - одинаковые `ROC-AUC`, `PR-AUC`, `num_users`, `positive_rate`.

3. Собрать финальный narrative:
   - RNN хорошо учит next-event структуру;
   - pooling и capacity важны;
   - рост next-event качества не гарантирует лучший retention;
   - финальный RNN-кандидат даёт устойчивый, но умеренный downstream lift;
   - следующее честное сравнение должно быть только в общем evaluator.

Сделано: ключевые RNN-серии вынесены в `scripts/run_rnn_experiments.py`. Раннер не переобучает тяжелые модели с нуля, но по `configs/rnn_experiments.yaml` валидирует и материализует таблицы, графики и manifest для финальных артефактов.
