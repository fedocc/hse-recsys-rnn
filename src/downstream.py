from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import re

import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import PreparedSequences


SECONDS_IN_DAY = 86_400


def slugify(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def build_base_meta(
    clean_events_glob: str,
    user_ids: Sequence[str],
    *,
    user_col: str = "appmetrica_device_id",
) -> tuple[pl.DataFrame, int]:
    meta = (
        pl.scan_parquet(clean_events_glob)
        .select(
            [
                pl.col(user_col).cast(pl.String),
                pl.col("event_timestamp").cast(pl.Int64),
                pl.col("session_id").cast(pl.Int64, strict=False),
                pl.col("row_id").cast(pl.Int64),
            ]
        )
        .sort([user_col, "event_timestamp", "row_id"])
        .group_by(user_col, maintain_order=True)
        .agg(
            [
                pl.col("event_timestamp").alias("all_timestamps"),
                pl.col("session_id").alias("all_sessions"),
                pl.col("event_timestamp").last().alias("last_event_ts"),
            ]
        )
        .collect()
        .filter(pl.col(user_col).is_in(list(user_ids)))
    )
    return meta, int(meta["last_event_ts"].max())


def build_meta_for_prefix(
    base_meta: pl.DataFrame,
    *,
    prefix_len: int,
    horizons: dict[str, int],
    data_end_ts: int,
    user_col: str = "appmetrica_device_id",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    meta = (
        base_meta.with_columns(
            [
                pl.col("all_timestamps").list.slice(0, prefix_len).alias("prefix_timestamps"),
                pl.col("all_sessions").list.slice(0, prefix_len).alias("prefix_sessions"),
            ]
        )
        .with_columns(
            [
                pl.col("prefix_timestamps").list.len().alias("prefix_len_time"),
                pl.col("prefix_timestamps").list.last().alias("prefix_end_ts"),
                pl.col("prefix_sessions").list.n_unique().alias("prefix_session_count"),
                pl.col("prefix_timestamps")
                .list.eval(pl.element() // SECONDS_IN_DAY)
                .list.n_unique()
                .alias("prefix_active_days"),
            ]
        )
    )

    for label_name, horizon_days in horizons.items():
        horizon_seconds = horizon_days * SECONDS_IN_DAY
        meta = meta.with_columns(
            pl.when((pl.col("prefix_len_time") == prefix_len) & (pl.col("prefix_end_ts") + horizon_seconds <= data_end_ts))
            .then((pl.col("last_event_ts") >= pl.col("prefix_end_ts") + horizon_seconds).cast(pl.Int8))
            .otherwise(None)
            .alias(label_name)
        )

    summary_rows = []
    for label_name in horizons:
        valid = meta.filter(pl.col(label_name).is_not_null())
        summary_rows.append(
            {
                "prefix_len": prefix_len,
                "label": label_name,
                "valid_users": valid.height,
                "positive_rate": float(valid[label_name].mean()) if valid.height else None,
            }
        )
    return meta, pl.DataFrame(summary_rows)


def build_meta_for_full_history(
    base_meta: pl.DataFrame,
    *,
    horizons: dict[str, int],
    user_col: str = "appmetrica_device_id",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    meta = base_meta.with_columns(
        [
            pl.col("all_timestamps").list.len().alias("history_len"),
            pl.col("all_timestamps").list.first().alias("first_event_ts"),
            pl.col("all_timestamps").list.last().alias("history_end_ts"),
            pl.col("all_sessions").list.n_unique().alias("history_session_count"),
            pl.col("all_timestamps")
            .list.eval(pl.element() // SECONDS_IN_DAY)
            .list.n_unique()
            .alias("history_active_days"),
        ]
    ).with_columns(
        ((pl.col("history_end_ts") - pl.col("first_event_ts")) / SECONDS_IN_DAY).alias("history_span_days")
    )

    for label_name, horizon_days in horizons.items():
        meta = meta.with_columns((pl.col("history_span_days") >= horizon_days).cast(pl.Int8).alias(label_name))

    summary_rows = []
    for label_name in horizons:
        valid = meta.filter(pl.col(label_name).is_not_null())
        summary_rows.append(
            {
                "history_window": "full",
                "label": label_name,
                "valid_users": valid.height,
                "positive_rate": float(valid[label_name].mean()) if valid.height else None,
            }
        )
    return meta.select(
        [
            user_col,
            "all_timestamps",
            "all_sessions",
            "last_event_ts",
            "history_len",
            "first_event_ts",
            "history_end_ts",
            "history_session_count",
            "history_active_days",
            "history_span_days",
            *horizons.keys(),
        ]
    ), pl.DataFrame(summary_rows)


def build_baseline_features(
    prepared: PreparedSequences,
    train_prepared: PreparedSequences,
    split_users: pl.DataFrame,
    meta: pl.DataFrame,
    *,
    prefix_len: int,
    top_k_events: int,
    horizons: dict[str, int],
    user_col: str = "appmetrica_device_id",
) -> tuple[pl.DataFrame, list[str], pl.DataFrame]:
    id2event = {idx: event for event, idx in prepared.event2idx.items()}
    train_prefix_counter: Counter[int] = Counter()
    for seq in train_prepared.user_sequences:
        train_prefix_counter.update(seq[:prefix_len])

    top_event_ids = [event_id for event_id, _ in train_prefix_counter.most_common(top_k_events)]
    top_event_names = [id2event[event_id] for event_id in top_event_ids]
    top_event_feature_names = [f"share_{slugify(name)}" for name in top_event_names]

    rows = []
    for user_id, seq in zip(prepared.user_ids, prepared.user_sequences, strict=True):
        prefix_seq = seq[:prefix_len]
        actual_len = len(prefix_seq)
        prefix_counter = Counter(prefix_seq)
        unique_event_count = len(prefix_counter)
        repeat_ratio = 1.0 - unique_event_count / actual_len if actual_len else 0.0

        row = {
            user_col: user_id,
            "prefix_len_feature": actual_len,
            "unique_event_count": unique_event_count,
            "repeat_ratio": repeat_ratio,
        }
        for feature_name, event_id in zip(top_event_feature_names, top_event_ids, strict=True):
            row[feature_name] = prefix_counter.get(event_id, 0) / max(actual_len, 1)
        rows.append(row)

    feature_df = (
        pl.DataFrame(rows)
        .join(
            meta.select([user_col, "prefix_session_count", "prefix_active_days", *horizons.keys()]),
            on=user_col,
            how="inner",
        )
        .join(split_users, on=user_col, how="inner")
    )

    feature_cols = [
        "prefix_len_feature",
        "unique_event_count",
        "repeat_ratio",
        "prefix_session_count",
        "prefix_active_days",
        *top_event_feature_names,
    ]
    summary = pl.DataFrame(
        {
            "prefix_len": [prefix_len],
            "feature_group": ["baseline"],
            "num_features": [len(feature_cols)],
            "top_events_used": [", ".join(top_event_names)],
        }
    )
    return feature_df, feature_cols, summary


def build_full_history_baseline_features(
    prepared: PreparedSequences,
    train_prepared: PreparedSequences,
    split_users: pl.DataFrame,
    meta: pl.DataFrame,
    *,
    max_history_len: int,
    top_k_events: int,
    horizons: dict[str, int],
    user_col: str = "appmetrica_device_id",
) -> tuple[pl.DataFrame, list[str], pl.DataFrame]:
    id2event = {idx: event for event, idx in prepared.event2idx.items()}
    train_window_counter: Counter[int] = Counter()
    for seq in train_prepared.user_sequences:
        train_window_counter.update(seq[-max_history_len:])

    top_event_ids = [event_id for event_id, _ in train_window_counter.most_common(top_k_events)]
    top_event_names = [id2event[event_id] for event_id in top_event_ids]
    top_event_feature_names = [f"share_{slugify(name)}" for name in top_event_names]

    rows = []
    for user_id, seq in zip(prepared.user_ids, prepared.user_sequences, strict=True):
        history_seq = seq[-max_history_len:]
        actual_len = len(history_seq)
        history_counter = Counter(history_seq)
        unique_event_count = len(history_counter)
        repeat_ratio = 1.0 - unique_event_count / actual_len if actual_len else 0.0

        row = {
            user_col: user_id,
            "history_len_feature": actual_len,
            "unique_event_count": unique_event_count,
            "repeat_ratio": repeat_ratio,
        }
        for feature_name, event_id in zip(top_event_feature_names, top_event_ids, strict=True):
            row[feature_name] = history_counter.get(event_id, 0) / max(actual_len, 1)
        rows.append(row)

    feature_df = (
        pl.DataFrame(rows)
        .join(
            meta.select(
                [
                    user_col,
                    "history_session_count",
                    "history_active_days",
                    *horizons.keys(),
                ]
            ),
            on=user_col,
            how="inner",
        )
        .join(split_users, on=user_col, how="inner")
    )

    feature_cols = [
        "history_len_feature",
        "unique_event_count",
        "repeat_ratio",
        "history_session_count",
        "history_active_days",
        *top_event_feature_names,
    ]
    summary = pl.DataFrame(
        {
            "history_window": [f"last_{max_history_len}"],
            "feature_group": ["baseline"],
            "num_features": [len(feature_cols)],
            "top_events_used": [", ".join(top_event_names)],
        }
    )
    return feature_df, feature_cols, summary


def evaluate_xgboost_feature_set(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    feature_set_name: str,
    *,
    seed: int,
) -> pl.DataFrame:
    train_df = df.filter((pl.col("split") == "train") & pl.col(label_col).is_not_null())
    val_df = df.filter((pl.col("split") == "val") & pl.col(label_col).is_not_null())
    test_df = df.filter((pl.col("split") == "test") & pl.col(label_col).is_not_null())

    x_train = train_df.select(list(feature_cols)).to_numpy()
    y_train = train_df[label_col].to_numpy().astype(int)

    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        verbosity=0,
    )
    model.fit(x_train, y_train)

    rows = []
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        x_split = split_df.select(list(feature_cols)).to_numpy()
        y_split = split_df[label_col].to_numpy().astype(int)
        y_prob = model.predict_proba(x_split)[:, 1]
        rows.append(
            {
                "feature_set": feature_set_name,
                "label": label_col,
                "split": split_name,
                "roc_auc": float(roc_auc_score(y_split, y_prob)),
                "pr_auc": float(average_precision_score(y_split, y_prob)),
                "num_users": int(len(y_split)),
            }
        )
    return pl.DataFrame(rows)
