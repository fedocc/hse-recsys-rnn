from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import random

import polars as pl

from .data import PreparedSequences


def build_random_embeddings(
    user_ids: Sequence[str],
    *,
    embedding_dim: int,
    seed: int = 42,
    user_col: str = "appmetrica_device_id",
) -> pl.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for user_id in user_ids:
        row = {user_col: user_id}
        row.update({f"emb_{idx:03d}": rng.gauss(0.0, 1.0) for idx in range(embedding_dim)})
        rows.append(row)
    return pl.DataFrame(rows)


def shuffle_user_sequences(
    prepared: PreparedSequences,
    *,
    seed: int = 42,
) -> PreparedSequences:
    rng = random.Random(seed)
    shuffled_sequences: list[list[int]] = []
    for seq in prepared.user_sequences:
        shuffled = list(seq)
        rng.shuffle(shuffled)
        shuffled_sequences.append(shuffled)
    return PreparedSequences(
        user_ids=list(prepared.user_ids),
        user_sequences=shuffled_sequences,
        event2idx=dict(prepared.event2idx),
    )


def shuffle_user_prefixes(
    prepared: PreparedSequences,
    *,
    prefix_len: int,
    seed: int = 42,
) -> PreparedSequences:
    rng = random.Random(seed)
    shuffled_sequences: list[list[int]] = []
    for seq in prepared.user_sequences:
        prefix = list(seq[:prefix_len])
        rng.shuffle(prefix)
        shuffled_sequences.append(prefix)
    return PreparedSequences(
        user_ids=list(prepared.user_ids),
        user_sequences=shuffled_sequences,
        event2idx=dict(prepared.event2idx),
    )


def build_bag_of_events_features(
    prepared: PreparedSequences,
    *,
    event_ids: Sequence[int],
    prefix_len: int,
    user_col: str = "appmetrica_device_id",
) -> tuple[pl.DataFrame, list[str]]:
    feature_cols = [f"event_share_{event_id:03d}" for event_id in event_ids]
    rows: list[dict[str, object]] = []

    for user_id, seq in zip(prepared.user_ids, prepared.user_sequences, strict=True):
        prefix = seq[:prefix_len]
        counts = Counter(prefix)
        denom = max(1, len(prefix))
        row = {user_col: user_id}
        for feature_col, event_id in zip(feature_cols, event_ids, strict=True):
            row[feature_col] = counts.get(event_id, 0) / denom
        rows.append(row)

    return pl.DataFrame(rows), feature_cols
