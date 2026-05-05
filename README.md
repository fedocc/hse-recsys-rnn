# Пользовательские эмбеддинги с помощью RNN

Курсовой проект в статусе `work in progress`: построение пользовательских эмбеддингов по последовательностям игровых событий с помощью рекуррентных нейронных сетей.

Главная тема моей части курсача — проверить, насколько `GRU/LSTM`-модели могут сжимать историю действий пользователя в полезный вектор. Для этого в проекте собран pipeline: preprocessing логов, построение пользовательских последовательностей, обучение RNN на next-event prediction, экспорт эмбеддингов и downstream-оценка на задаче retention.

В финальной командной работе RNN-подход планируется сравнить с более сложными sequence-моделями, но этот репозиторий в первую очередь про RNN-based user embeddings.

## Текущий статус

Это ранняя исследовательская версия репозитория, не финальная сдача курсача.

Уже реализовано:

- очистка event logs и построение последовательностей пользователей;
- фиксированный `train/val/test` split по пользователям;
- простые sequence-baseline: `MostPopular` и `Markov-1`;
- `GRU` и `LSTM` модели для next-event prediction;
- оценка через `Hit@5`, `Hit@10`, `MRR`;
- экспорт пользовательских эмбеддингов;
- downstream benchmark для retention с hand-crafted признаками и RNN-эмбеддингами.

В работе:

- ablation-исследование того, как длина истории, pooling и capacity RNN влияют на качество эмбеддингов;
- финальное сравнение RNN-подхода с другими sequence-моделями в общей командной работе.

## Структура проекта

```text
hse_recsys_rnn/
  src/
    data.py        # очистка, построение последовательностей, Dataset
    splits.py      # user-level train/val/test split
    baselines.py   # MostPopular и Markov-1 baselines
    models.py      # GRU/LSTM encoder для user embeddings
    train.py       # train loop
    eval.py        # Hit@K / MRR evaluation
    export.py      # экспорт user embeddings
    pooling.py     # pooling hidden states
  notebooks/
    01_data_and_preprocessing.ipynb
    02_rnn_baseline.ipynb
    03_downstream_evaluation.ipynb
  configs/
    rnn_experiments.yaml
  reports/
    project_deep_dive.md
    rnn_next_steps.md
```

Большие сырые данные, intermediate parquet-таблицы и user-level split-файлы намеренно не коммитятся.

## Текущие результаты

### RNN sequence modeling

Текущая оценка next-event prediction на фиксированном split:

| model | split | MRR | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|
| GRU | test | 0.8804 | 0.9686 | 0.9910 |
| LSTM | test | 0.8806 | 0.9689 | 0.9911 |
| Markov-1 | test | 0.8128 | 0.9395 | 0.9765 |
| MostPopular | test | 0.4594 | 0.7278 | 0.8910 |

RNN-модели уже уверенно обгоняют popularity и Markov baseline по sequence-метрикам, то есть лучше восстанавливают локальную структуру пользовательских событий.

### Downstream retention

Для `prefix_len = 150` текущий downstream benchmark на `retention_14d`:

| feature set | split | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Baseline | test | 0.8741 | 0.3142 |
| Baseline + GRU | test | 0.8762 | 0.3216 |
| Baseline + LSTM | test | 0.8720 | 0.3043 |
| GRU only | test | 0.8462 | 0.2764 |
| LSTM only | test | 0.8372 | 0.2525 |

Результат пока предварительный: RNN-эмбеддинги сами по себе уже несут сигнал, но на текущей итерации лучше всего работает комбинация sequence representations с простыми hand-crafted признаками.

## Запуск

Установка зависимостей:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ноутбуки рассчитаны на запуск из корня репозитория:

```bash
jupyter notebook
```

В репозитории лежат небольшие summary artifacts и метрики. Полные сырые логи и большие embedding-таблицы должны храниться локально в `artifacts/` и игнорируются git.

## Следующие шаги

- провести context saturation experiments для `prefix_len = 25..300`;
- сравнить pooling strategies: `last`, `mean`, `max`, `last_mean`;
- вынести downstream utilities из ноутбука в `src/downstream.py`;
- проверить, как изменения RNN-архитектуры переносятся из next-event quality в downstream retention;
- после интеграции с командой сравнить RNN-подход с другими sequence-моделями и собрать финальный отчёт по курсачу.
