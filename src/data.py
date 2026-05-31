from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset


TECH_EVENTS = {
    "BOOT_TIME",
    "BOOT_TIME_SECONDS",
    "AB failed to initialize",
    "SystemInitialized",
    "SystemInitializing",
    "Game loaded",
}

EXCLUDED_EVENTS = {
    "PlayTimeMinutes",
    "PlayTimeAfkMinutes",
    "AB",
    "CohortData",
    "GameData",
    "Replay",
    "DebugAdEvents",
}

EXCLUDED_PREFIXES = ("__", "debug_", "test_", "internal_")


@dataclass
class PreparedSequences:
    user_ids: list[str]
    user_sequences: list[list[int]]
    event2idx: dict[str, int]


@dataclass(frozen=True)
class NextEventExample:
    prefix: list[int]
    target: int


class EventWindowDataset(Dataset):
    def __init__(self, user_sequences: list[list[int]], max_len: int, stride: int):
        self.max_tokens = max_len + 1
        self.samples: list[list[int]] = []

        for seq in user_sequences:
            if len(seq) < 2:
                continue

            starts = list(range(0, max(1, len(seq) - 1), stride))
            last_start = max(0, len(seq) - self.max_tokens)
            starts.append(last_start)

            for start in sorted(set(starts)):
                window = seq[start : start + self.max_tokens]
                if len(window) >= 2:
                    self.samples.append(window)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.samples[idx]
        length = len(seq)
        padded = seq + [0] * (self.max_tokens - length)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(length, dtype=torch.long)


class LastEventWindowDataset(Dataset):
    def __init__(self, user_sequences: list[list[int]], max_len: int):
        self.max_tokens = max_len + 1
        self.samples: list[list[int]] = []

        for seq in user_sequences:
            if len(seq) < 2:
                continue
            self.samples.append(seq[-self.max_tokens :])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.samples[idx]
        length = len(seq)
        padded = seq + [0] * (self.max_tokens - length)
        return torch.tensor(padded, dtype=torch.long), torch.tensor(length, dtype=torch.long)


def prepared_sequences_to_frame(
    prepared: PreparedSequences,
    *,
    user_col: str = "appmetrica_device_id",
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            user_col: prepared.user_ids,
            "sequence": prepared.user_sequences,
            "sequence_len": [len(seq) for seq in prepared.user_sequences],
        }
    )


def _exclude_event_expr(
    event_col: str,
    excluded_events: set[str],
    excluded_prefixes: tuple[str, ...],
) -> pl.Expr:
    prefix_expr = pl.lit(False)
    for prefix in excluded_prefixes:
        prefix_expr = prefix_expr | pl.col(event_col).str.starts_with(prefix)
    return pl.col(event_col).is_in(sorted(excluded_events)) | prefix_expr


def _normalize_event_chunk(
    df: pl.DataFrame,
    *,
    row_id_offset: int,
    user_col: str,
    event_col: str,
    timestamp_col: str,
    excluded_events: set[str],
    excluded_prefixes: tuple[str, ...],
) -> pl.DataFrame:
    if df.is_empty():
        return df

    normalized = (
        df.with_row_index("row_in_chunk", offset=row_id_offset)
        .rename({"row_in_chunk": "row_id"})
        .drop_nulls(subset=[user_col, event_col, timestamp_col])
        .with_columns(
            pl.col(user_col).cast(pl.String),
            pl.col(event_col).cast(pl.String),
            pl.col(timestamp_col).cast(pl.Int64, strict=False),
        )
        .drop_nulls(subset=[timestamp_col])
        .filter(~_exclude_event_expr(event_col, excluded_events, excluded_prefixes))
        .sort([user_col, timestamp_col, "row_id"])
    )
    return normalized


def preprocess_event_logs_to_parquet(
    csv_path: Path | str,
    out_dir: Path | str,
    *,
    chunk_size: int = 250_000,
    user_col: str = "appmetrica_device_id",
    event_col: str = "event_name",
    timestamp_col: str = "event_timestamp",
    usecols: list[str] | None = None,
    excluded_events: set[str] | None = None,
    excluded_prefixes: tuple[str, ...] = EXCLUDED_PREFIXES,
    overwrite: bool = False,
) -> dict[str, Path]:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    parts_dir = out_dir / "clean_events"

    if overwrite and parts_dir.exists():
        for part in parts_dir.glob("*.parquet"):
            part.unlink()

    parts_dir.mkdir(parents=True, exist_ok=True)

    if usecols is None:
        usecols = [
            user_col,
            event_col,
            "event_json",
            "event_datetime",
            timestamp_col,
            "session_id",
        ]

    events_to_drop = set(TECH_EVENTS) | set(EXCLUDED_EVENTS)
    if excluded_events is not None:
        events_to_drop |= set(excluded_events)

    total_rows_read = 0
    total_rows_kept = 0
    part_paths: list[Path] = []

    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size)
    for part_idx, chunk in enumerate(reader):
        total_rows_read += len(chunk)
        part_df = pl.DataFrame(chunk.to_dict(orient="list"), strict=False)
        cleaned = _normalize_event_chunk(
            part_df,
            row_id_offset=total_rows_read - len(chunk),
            user_col=user_col,
            event_col=event_col,
            timestamp_col=timestamp_col,
            excluded_events=events_to_drop,
            excluded_prefixes=excluded_prefixes,
        )

        total_rows_kept += cleaned.height
        part_path = parts_dir / f"part_{part_idx:04d}.parquet"
        cleaned.write_parquet(part_path)
        part_paths.append(part_path)

    summary = pl.DataFrame(
        {
            "metric": [
                "raw_csv_path",
                "chunk_size",
                "parts_written",
                "rows_read",
                "rows_kept",
            ],
            "value": pl.Series(
                "value",
                [
                    str(csv_path),
                    chunk_size,
                    len(part_paths),
                    total_rows_read,
                    total_rows_kept,
                ],
                strict=False,
            ),
        }
    )
    summary_path = out_dir / "clean_events_summary.csv"
    summary.write_csv(summary_path)

    return {
        "parts_dir": parts_dir,
        "summary": summary_path,
    }


def load_clean_event_logs(
    clean_dir: Path | str,
    *,
    user_col: str = "appmetrica_device_id",
    timestamp_col: str = "event_timestamp",
) -> pl.DataFrame:
    clean_dir = Path(clean_dir)
    part_paths = sorted(clean_dir.glob("*.parquet"))
    if not part_paths:
        raise FileNotFoundError(f"No parquet parts found in {clean_dir}")

    df = pl.concat([pl.read_parquet(path) for path in part_paths], how="vertical_relaxed")
    return df.sort([user_col, timestamp_col, "row_id"])


def load_event_logs(
    csv_path: Path | str,
    *,
    user_col: str = "appmetrica_device_id",
    event_col: str = "event_name",
    timestamp_col: str = "event_timestamp",
    excluded_events: set[str] | None = None,
    excluded_prefixes: tuple[str, ...] = EXCLUDED_PREFIXES,
) -> pl.DataFrame:
    events_to_drop = set(TECH_EVENTS) | set(EXCLUDED_EVENTS)
    if excluded_events is not None:
        events_to_drop |= set(excluded_events)

    df = pl.read_csv(csv_path, schema_overrides={user_col: pl.String})
    return _normalize_event_chunk(
        df,
        row_id_offset=0,
        user_col=user_col,
        event_col=event_col,
        timestamp_col=timestamp_col,
        excluded_events=events_to_drop,
        excluded_prefixes=excluded_prefixes,
    )


def summarize_event_logs(
    df: pl.DataFrame,
    *,
    user_col: str = "appmetrica_device_id",
    event_col: str = "event_name",
    timestamp_col: str = "event_timestamp",
    session_col: str = "session_id",
) -> pl.DataFrame:
    metrics = {
        "rows": df.height,
        "users": df[user_col].n_unique(),
        "unique_events": df[event_col].n_unique(),
        "min_timestamp": df[timestamp_col].min(),
        "max_timestamp": df[timestamp_col].max(),
    }
    if session_col in df.columns:
        metrics["sessions"] = df[session_col].n_unique()
    return pl.DataFrame(
        {
            "metric": list(metrics.keys()),
            "value": pl.Series("value", list(metrics.values()), strict=False),
        }
    )


def build_event_vocabulary(df: pl.DataFrame, *, event_col: str = "event_name") -> dict[str, int]:
    event_names = sorted(df.select(event_col).unique().to_series().to_list())
    return {name: idx + 1 for idx, name in enumerate(event_names)}


def build_user_sequences(
    df: pl.DataFrame,
    *,
    min_events_per_user: int = 10,
    user_col: str = "appmetrica_device_id",
    event_col: str = "event_name",
    event2idx: dict[str, int] | None = None,
) -> PreparedSequences:
    vocab = event2idx or build_event_vocabulary(df, event_col=event_col)
    grouped = (
        df.with_columns(pl.col(event_col).replace(vocab).cast(pl.Int64).alias("event_idx"))
        .group_by(user_col, maintain_order=True)
        .agg(pl.col("event_idx").alias("seq"))
        .filter(pl.col("seq").list.len() >= min_events_per_user)
    )

    return PreparedSequences(
        user_ids=grouped[user_col].to_list(),
        user_sequences=grouped["seq"].to_list(),
        event2idx=vocab,
    )


def summarize_sequence_lengths(prepared: PreparedSequences) -> pl.DataFrame:
    lengths = [len(seq) for seq in prepared.user_sequences]
    if not lengths:
        return pl.DataFrame({"metric": ["users"], "value": pl.Series("value", [0], strict=False)})

    stats = {
        "users": len(lengths),
        "vocab_size": len(prepared.event2idx),
        "min_len": min(lengths),
        "q25_len": int(pl.Series(lengths).quantile(0.25, interpolation="nearest")),
        "median_len": int(pl.Series(lengths).median()),
        "q75_len": int(pl.Series(lengths).quantile(0.75, interpolation="nearest")),
        "p95_len": int(pl.Series(lengths).quantile(0.95, interpolation="nearest")),
        "max_len": max(lengths),
        "mean_len": round(sum(lengths) / len(lengths), 2),
    }
    return pl.DataFrame(
        {
            "metric": list(stats.keys()),
            "value": pl.Series("value", list(stats.values()), strict=False),
        }
    )


def save_prepared_sequences(
    prepared: PreparedSequences,
    out_dir: Path | str,
    *,
    user_col: str = "appmetrica_device_id",
) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sequences_path = out_path / "user_sequences.parquet"
    vocab_path = out_path / "event2idx.json"
    summary_path = out_path / "sequence_summary.csv"

    prepared_sequences_to_frame(prepared, user_col=user_col).write_parquet(sequences_path)
    vocab_path.write_text(json.dumps(prepared.event2idx, ensure_ascii=False, indent=2), encoding="utf-8")
    summarize_sequence_lengths(prepared).write_csv(summary_path)

    return {
        "user_sequences": sequences_path,
        "event2idx": vocab_path,
        "sequence_summary": summary_path,
    }


def load_prepared_sequences(
    in_dir: Path | str,
    *,
    user_col: str = "appmetrica_device_id",
) -> PreparedSequences:
    in_path = Path(in_dir)
    sequences_df = pl.read_parquet(in_path / "user_sequences.parquet")
    event2idx = json.loads((in_path / "event2idx.json").read_text(encoding="utf-8"))

    return PreparedSequences(
        user_ids=sequences_df[user_col].to_list(),
        user_sequences=sequences_df["sequence"].to_list(),
        event2idx=event2idx,
    )


def load_sequences(
    csv_path: Path | str,
    min_events_per_user: int,
    *,
    user_col: str = "appmetrica_device_id",
    event_col: str = "event_name",
    timestamp_col: str = "event_timestamp",
) -> PreparedSequences:
    df = load_event_logs(
        csv_path,
        user_col=user_col,
        event_col=event_col,
        timestamp_col=timestamp_col,
    )
    return build_user_sequences(
        df,
        min_events_per_user=min_events_per_user,
        user_col=user_col,
        event_col=event_col,
    )


def build_next_event_examples(
    user_sequences: list[list[int]],
    max_len: int,
    stride: int,
) -> list[NextEventExample]:
    dataset = EventWindowDataset(user_sequences, max_len=max_len, stride=stride)
    return [NextEventExample(prefix=window[:-1], target=window[-1]) for window in dataset.samples]


def build_last_event_examples(
    user_sequences: list[list[int]],
    max_len: int,
) -> list[NextEventExample]:
    dataset = LastEventWindowDataset(user_sequences, max_len=max_len)
    return [NextEventExample(prefix=window[:-1], target=window[-1]) for window in dataset.samples]
