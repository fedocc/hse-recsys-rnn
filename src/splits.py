from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split

from .data import PreparedSequences


@dataclass
class UserSplit:
    train_user_ids: list[str]
    val_user_ids: list[str]
    test_user_ids: list[str]
    seed: int
    val_ratio: float
    test_ratio: float


def build_user_split(
    user_ids: list[str],
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> UserSplit:
    if not user_ids:
        raise ValueError("Cannot split an empty user list.")
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1.")

    temp_ratio = val_ratio + test_ratio
    train_user_ids, temp_user_ids = train_test_split(
        list(user_ids),
        test_size=temp_ratio,
        random_state=seed,
    )
    val_user_ids, test_user_ids = train_test_split(
        temp_user_ids,
        test_size=test_ratio / temp_ratio,
        random_state=seed,
    )

    return UserSplit(
        train_user_ids=list(train_user_ids),
        val_user_ids=list(val_user_ids),
        test_user_ids=list(test_user_ids),
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def build_user_split_from_master_csv(
    user_ids: list[str],
    master_split_path: Path | str,
    *,
    seed: int = 42,
    user_col: str = "appmetrica_device_id",
    split_col: str = "split",
) -> tuple[UserSplit, pl.DataFrame]:
    master_split_path = Path(master_split_path)
    with master_split_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        expected_columns = [user_col, split_col]
        if reader.fieldnames != expected_columns:
            raise ValueError(f"Expected columns {expected_columns}, got {reader.fieldnames}")
        rows = [{user_col: str(row[user_col]), split_col: row[split_col]} for row in reader]

    allowed_splits = {"train", "val", "test"}
    bad_splits = sorted({row[split_col] for row in rows} - allowed_splits)
    if bad_splits:
        raise ValueError(f"Unsupported split values: {bad_splits}")

    master_user_ids = [row[user_col] for row in rows]
    duplicates = len(master_user_ids) - len(set(master_user_ids))
    if duplicates:
        raise ValueError(f"Master split contains duplicate user ids: {duplicates}")

    prepared_user_ids = set(map(str, user_ids))
    master_user_id_set = set(master_user_ids)
    split_to_users = {"train": [], "val": [], "test": []}
    for row in rows:
        user_id = row[user_col]
        if user_id in prepared_user_ids:
            split_to_users[row[split_col]].append(user_id)

    split = UserSplit(
        train_user_ids=split_to_users["train"],
        val_user_ids=split_to_users["val"],
        test_user_ids=split_to_users["test"],
        seed=seed,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    summary = pl.DataFrame(
        [
            {"metric": "master_split_path", "value": str(master_split_path)},
            {"metric": "master_users", "value": len(master_user_id_set)},
            {"metric": "prepared_users", "value": len(prepared_user_ids)},
            {"metric": "overlap_users", "value": len(prepared_user_ids & master_user_id_set)},
            {"metric": "prepared_users_missing_in_master", "value": len(prepared_user_ids - master_user_id_set)},
            {"metric": "master_users_missing_in_prepared", "value": len(master_user_id_set - prepared_user_ids)},
            {"metric": "rnn_train_users", "value": len(split.train_user_ids)},
            {"metric": "rnn_val_users", "value": len(split.val_user_ids)},
            {"metric": "rnn_test_users", "value": len(split.test_user_ids)},
        ]
    )
    return split, summary


def save_user_split(split: UserSplit, path: Path | str) -> None:
    payload = {
        "seed": split.seed,
        "val_ratio": split.val_ratio,
        "test_ratio": split.test_ratio,
        "train_user_ids": split.train_user_ids,
        "val_user_ids": split.val_user_ids,
        "test_user_ids": split.test_user_ids,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_user_split_tables(
    prepared: PreparedSequences,
    split: UserSplit,
    out_dir: Path | str,
) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    split_json_path = out_path / "split.json"
    split_summary_path = out_path / "split_summary.csv"
    split_users_path = out_path / "split_users.parquet"

    save_user_split(split, split_json_path)
    describe_user_split(prepared, split).write_csv(split_summary_path)

    rows: list[dict[str, str]] = []
    for split_name, user_ids in (
        ("train", split.train_user_ids),
        ("val", split.val_user_ids),
        ("test", split.test_user_ids),
    ):
        for user_id in user_ids:
            rows.append({"appmetrica_device_id": user_id, "split": split_name})

    pl.DataFrame(rows).write_parquet(split_users_path)
    return {
        "split_json": split_json_path,
        "split_summary": split_summary_path,
        "split_users": split_users_path,
    }


def load_user_split(path: Path | str) -> UserSplit:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return UserSplit(
        train_user_ids=payload["train_user_ids"],
        val_user_ids=payload["val_user_ids"],
        test_user_ids=payload["test_user_ids"],
        seed=payload["seed"],
        val_ratio=payload["val_ratio"],
        test_ratio=payload["test_ratio"],
    )


def apply_user_split(prepared: PreparedSequences, split: UserSplit) -> dict[str, PreparedSequences]:
    user_to_seq = dict(zip(prepared.user_ids, prepared.user_sequences, strict=True))

    def subset(user_ids: list[str]) -> PreparedSequences:
        subset_user_ids = [user_id for user_id in user_ids if user_id in user_to_seq]
        return PreparedSequences(
            user_ids=subset_user_ids,
            user_sequences=[user_to_seq[user_id] for user_id in subset_user_ids],
            event2idx=prepared.event2idx,
        )

    return {
        "train": subset(split.train_user_ids),
        "val": subset(split.val_user_ids),
        "test": subset(split.test_user_ids),
    }


def describe_user_split(prepared: PreparedSequences, split: UserSplit) -> pl.DataFrame:
    lengths = dict(zip(prepared.user_ids, [len(seq) for seq in prepared.user_sequences], strict=True))

    rows = []
    for split_name, user_ids in (
        ("train", split.train_user_ids),
        ("val", split.val_user_ids),
        ("test", split.test_user_ids),
    ):
        split_lengths = [lengths[user_id] for user_id in user_ids if user_id in lengths]
        rows.append(
            {
                "split": split_name,
                "users": len(user_ids),
                "mean_seq_len": round(sum(split_lengths) / len(split_lengths), 2) if split_lengths else 0.0,
                "median_seq_len": float(pl.Series(split_lengths).median()) if split_lengths else 0.0,
                "max_seq_len": max(split_lengths) if split_lengths else 0,
            }
        )

    return pl.DataFrame(rows)
