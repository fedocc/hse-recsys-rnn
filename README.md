# Пользовательские эмбеддинги игровых событий на основе RNN

Репозиторий содержит RNN-часть курсового проекта по теме «Построение эмбеддингов пользователей по последовательности игровых событий на основе контрастивного обучения, трансформеров и RNN».

В этой ветке реализован метод RNN: подготовка последовательностей игровых событий, обучение GRU/LSTM на задаче next-event prediction, экспорт пользовательских эмбеддингов и downstream-оценка на retention-задачах. Для командного сравнения также добавлен согласованный full-history прогон RNN, сопоставимый с SimCLR и BERT4Rec по общей постановке.

## Главные материалы

- Финальный отчет: `reports/final_rnn_course_report.pdf`
- Исходник отчета: `reports/final_rnn_course_report.md`
- Основные ноутбуки: `notebooks/01_data_and_preprocessing.ipynb` - `notebooks/12_rnn_full_history.ipynb`
- Код моделей и оценки: `src/`
- Воспроизводимые скрипты: `scripts/`
- Конфигурация серий экспериментов: `configs/rnn_experiments.yaml`
- Сводная таблица RNN-результатов: `artifacts/results_summary.csv`

## Что реализовано

- Очистка логов и построение пользовательских последовательностей.
- User-level train/val/test split.
- Baseline для next-event prediction: MostPopular и Markov-1.
- GRU и LSTM encoder для построения пользовательских эмбеддингов.
- Оценка next-event prediction через MRR, Hit@5 и Hit@10.
- Prefix-based downstream retention evaluation без утечки будущих событий.
- Ablation по длине префикса, pooling-стратегии и capacity RNN.
- Supervised fine-tuning и negative controls.
- Анализ embedding space через нормы, PCA и KMeans.
- Full-history RNN-прогон для командного сравнения с SimCLR и BERT4Rec.
- Runner для проверки и материализации ключевых экспериментальных артефактов.

## Ключевые результаты

### Next-event prediction

| Model | Split | MRR | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|
| MostPopular | test | 0.4510 | 0.7154 | 0.8781 |
| Markov-1 | test | 0.8158 | 0.9408 | 0.9775 |
| GRU | test | 0.8885 | 0.9733 | 0.9926 |
| LSTM | test | 0.8886 | 0.9732 | 0.9927 |

### Prefix-based retention

Финальный RNN-кандидат: GRU h256, 1 layer, max pooling, prefix_len = 150.

| Target | Feature set | ROC-AUC | PR-AUC | Test users |
|---|---|---:|---:|---:|
| retention_7d | Baseline | 0.6609 ± 0.0022 | 0.4551 ± 0.0042 | 2,805 |
| retention_7d | Baseline + GRU h256 l1 max | 0.6922 ± 0.0052 | 0.4978 ± 0.0023 | 2,805 |
| retention_14d | Baseline | 0.6788 ± 0.0027 | 0.3412 ± 0.0043 | 2,303 |
| retention_14d | Baseline + GRU h256 l1 max | 0.7051 ± 0.0032 | 0.3644 ± 0.0037 | 2,303 |

### Full-history командное сравнение

В командной таблице сравниваются эмбеддинги без дополнительных baseline-признаков в общей full-history постановке.

| Horizon | Model | ROC-AUC | PR-AUC | Test users | Positive rate |
|---|---|---:|---:|---:|---:|
| 7d | BERT mean | 0.9296 | 0.6228 | 10,260 | 11.1% |
| 7d | RNN/GRU | 0.9413 | 0.6777 | 10,260 | 11.1% |
| 7d | SimCLR | 0.9249 | 0.6263 | 10,260 | 11.1% |
| 14d | BERT mean | 0.9256 | 0.4467 | 10,260 | 6.0% |
| 14d | RNN/GRU | 0.9319 | 0.4960 | 10,260 | 6.0% |
| 14d | SimCLR | 0.9189 | 0.4974 | 10,260 | 6.0% |
| 30d | BERT mean | 0.9665 | 0.1500 | 10,260 | 0.34% |
| 30d | RNN/GRU | 0.9658 | 0.1910 | 10,260 | 0.34% |
| 30d | SimCLR | 0.8621 | 0.1725 | 10,260 | 0.34% |

RNN-числа из этой таблицы воспроизводятся из `artifacts/rnn_full_history_simclr_aligned/aligned_summary.csv`. SimCLR и BERT приведены только для контекста командного сравнения в отчете.

## Структура репозитория

```text
hse_recsys_rnn/
  artifacts/      # метрики, summary-таблицы, графики и небольшие checkpoints
  configs/        # конфигурации экспериментов
  notebooks/      # исследовательские ноутбуки 01-12
  reports/        # финальный отчет и промежуточные заметки
  scripts/        # воспроизводимые runner/evaluation scripts
  src/            # код подготовки данных, моделей, pooling, downstream evaluation
  tests/          # unit tests для runner и full-history aligned pipeline
```

Большие промежуточные parquet-таблицы, сырые логи, системные кэши и временные smoke-embeddings не входят в git. В репозитории сохранены финальные метрики, графики, конфигурации, небольшие checkpoints, ноутбуки и отчет, то есть все материалы, необходимые для проверки выводов.

## Запуск и проверка

Установка зависимостей:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Сборка общей сводки результатов:

```bash
python scripts/build_results_summary.py
```

Проверка артефактов RNN-экспериментов:

```bash
python scripts/run_rnn_experiments.py --strict
```

Unit tests:

```bash
python -m unittest tests/test_run_rnn_experiments.py -v
python -m unittest tests/test_rnn_full_history_simclr_aligned.py -v
```

Сборка PDF-отчета из Markdown:

```bash
pandoc reports/final_rnn_course_report.md -o reports/final_rnn_course_report.pdf --pdf-engine=xelatex
```

## Ограничения

Проект является исследовательской веткой курсовой работы. Часть полной истории экспериментов сохранена в ноутбуках, а runner валидирует и материализует итоговые артефакты. Командное сравнение SimCLR/BERT/RNN приведено к общей full-history постановке, но ветки разрабатывались независимо.
