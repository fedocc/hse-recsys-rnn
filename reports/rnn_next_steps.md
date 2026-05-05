# Следующие шаги по RNN-ветке

Цель ближайшей итерации: превратить RNN-ветку из рабочего baseline в полноценное исследование того, как строить пользовательские эмбеддинги с помощью `GRU/LSTM`.

## Приоритетный порядок

1. **Каркас runner'а для экспериментов**
   - довести `scripts/run_rnn_experiments.py` до запуска реальных серий;
   - писать результаты в `artifacts/experiments/rnn/results.csv`;
   - не запускать эксперименты вручную из ноутбуков.

2. **Насыщение контекста**
   - `prefix_len = 25 / 50 / 75 / 100 / 150 / 200 / 300`;
   - метрики: `ROC-AUC`, `PR-AUC`, `valid_users`, `positive_rate`, `marginal_gain`;
   - главный вопрос: когда увеличение префикса перестаёт окупаться.

3. **Ablation по pooling**
   - `last`, `mean`, `max`, `last_mean`;
   - использовать одну обученную GRU как encoder;
   - главный вопрос: какой способ агрегировать RNN hidden states в user embedding лучше для retention.

4. **Перебор capacity модели**
   - `hidden_dim = 64 / 128 / 256`;
   - `max_len = 64 / 128 / 256`;
   - маленький controlled set, а не полный перебор;
   - главный вопрос: переносится ли рост sequence quality в downstream.

5. **Supervised GRU для retention**
   - отдельный supervised baseline;
   - не заменяет self-supervised next-event setup;
   - главный вопрос: стоит ли обучать RNN напрямую под retention или self-supervised next-event objective даёт более универсальные embeddings.

6. **Анализ эмбеддингов**
   - UMAP или KMeans profiles;
   - опционально, если останется время после основных абляций.

## Что уже подготовлено

- `configs/rnn_experiments.yaml` — карта экспериментов.
- `src/pooling.py` — функции для `last / mean / max / last_mean` pooling.
- `RecurrentEncoder.encode_steps(...)` — получение hidden states по всем шагам.
- `scripts/run_rnn_experiments.py` — стартовый каркас runner'а и структура output-директорий.

## Ближайшая техническая итерация

1. Проверить `src/pooling.py` на одной маленькой batch-выборке.
2. Вынести из `03_downstream_evaluation.ipynb` переиспользуемые downstream-функции в `src/downstream.py`.
3. Реализовать первую полноценную серию runner'а: `context_saturation`.
4. После первого полного прогона сохранить:
   - `context_saturation.csv`;
   - график `quality_vs_prefix`;
   - график `valid_users_vs_prefix`.
