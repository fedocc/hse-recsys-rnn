from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import LastEventWindowDataset, PreparedSequences, build_last_event_examples
from src.downstream import evaluate_xgboost_feature_set
from src.eval import evaluate_next_event_model_on_examples
from src.models import GRUEncoder, RecurrentEncoder
from src.pooling import build_user_embeddings_with_pooling
from src.train import choose_device, seed_everything, train_model


DEFAULT_SIMCLR_DIR = Path("/Users/fedornikonov/Downloads/курсач")
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "rnn_full_history_simclr_aligned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an RNN full-history branch on SimCLR-aligned last-512 sequences."
    )
    parser.add_argument("--simclr-dir", type=Path, default=DEFAULT_SIMCLR_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--history-len", type=int, default=512)
    parser.add_argument("--event-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=512)
    parser.add_argument("--pooling", choices=["last", "mean", "max", "last_mean"], default="max")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 77, 123, 202])
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="Train/export embeddings without XGBoost downstream.")
    parser.add_argument(
        "--max-users-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit. Full coursework runs should leave this unset.",
    )
    return parser.parse_args()


def strip_padding(sequence: list[int] | np.ndarray | torch.Tensor) -> list[int]:
    if isinstance(sequence, torch.Tensor):
        values = sequence.detach().cpu().tolist()
    elif isinstance(sequence, np.ndarray):
        values = sequence.tolist()
    else:
        values = list(sequence)
    return [int(value) for value in values if int(value) != 0]


def tensor_to_sequences(tensor: torch.Tensor) -> list[list[int]]:
    return [strip_padding(row) for row in tensor]


def load_split_sequences(simclr_dir: Path, *, max_users_per_split: int | None = None) -> dict[str, PreparedSequences]:
    processed_dir = simclr_dir / "data" / "processed"
    if not processed_dir.exists():
        raise FileNotFoundError(f"SimCLR processed directory not found: {processed_dir}")

    prepared_by_split: dict[str, PreparedSequences] = {}
    max_token_id = 0
    for split in ["train", "val", "test"]:
        sequences_tensor = torch.load(processed_dir / f"{split}_sequences.pt", map_location="cpu")
        user_ids = np.load(processed_dir / f"user_ids_{split}.npy", allow_pickle=True).astype(str).tolist()
        if max_users_per_split is not None:
            sequences_tensor = sequences_tensor[:max_users_per_split]
            user_ids = user_ids[:max_users_per_split]

        sequences = tensor_to_sequences(sequences_tensor)
        if len(user_ids) != len(sequences):
            raise ValueError(f"User ids and sequences length mismatch for split={split}")
        if sequences:
            max_token_id = max(max_token_id, max((max(seq, default=0) for seq in sequences), default=0))
        prepared_by_split[split] = PreparedSequences(user_ids=user_ids, user_sequences=sequences, event2idx={})

    event2idx = {f"token_{idx}": idx for idx in range(1, max_token_id + 1)}
    for prepared in prepared_by_split.values():
        prepared.event2idx.update(event2idx)
    return prepared_by_split


def combine_prepared(prepared_by_split: dict[str, PreparedSequences]) -> PreparedSequences:
    user_ids: list[str] = []
    sequences: list[list[int]] = []
    for split in ["train", "val", "test"]:
        prepared = prepared_by_split[split]
        user_ids.extend(prepared.user_ids)
        sequences.extend(prepared.user_sequences)
    return PreparedSequences(user_ids=user_ids, user_sequences=sequences, event2idx=prepared_by_split["train"].event2idx)


def build_split_frame(split_user_ids: dict[str, list[str]]) -> pl.DataFrame:
    rows = [
        {"appmetrica_device_id": user_id, "split": split}
        for split in ["train", "val", "test"]
        for user_id in split_user_ids[split]
    ]
    return pl.DataFrame(rows, schema={"appmetrica_device_id": pl.String, "split": pl.String})


def load_labels(simclr_dir: Path, split_frame: pl.DataFrame) -> pl.DataFrame:
    labels_path = simclr_dir / "data" / "export" / "retention_labels.csv"
    labels = pl.read_csv(labels_path, schema_overrides={"appmetrica_device_id": pl.String})
    return labels.join(split_frame, on="appmetrica_device_id", how="inner")


def load_baseline_features(simclr_dir: Path, labels_with_split: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    baseline_path = simclr_dir / "data" / "export" / "baseline_features.csv"
    baseline = pl.read_csv(baseline_path, schema_overrides={"appmetrica_device_id": pl.String})
    baseline_cols = [col for col in baseline.columns if col != "appmetrica_device_id"]
    return baseline.join(labels_with_split, on="appmetrica_device_id", how="inner"), baseline_cols


def make_model(args: argparse.Namespace, vocab_size: int, device: torch.device) -> RecurrentEncoder:
    return GRUEncoder(
        vocab_size=vocab_size,
        event_dim=args.event_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)


def train_or_load_model(
    args: argparse.Namespace,
    train_prepared: PreparedSequences,
    val_prepared: PreparedSequences,
    test_prepared: PreparedSequences,
    vocab_size: int,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RecurrentEncoder, dict[str, object], pl.DataFrame, pl.DataFrame]:
    config = {
        "experiment_id": f"simclr_aligned_gru_h{args.hidden_dim}_l{args.num_layers}_last{args.history_len}",
        "model_type": "gru",
        "event_dim": args.event_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "max_len": args.history_len,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_batch_size": args.train_batch_size,
        "vocab_size": vocab_size,
        "seed": 42,
        "source_sequences": "SimCLR data/processed/*_sequences.pt",
    }

    model = make_model(args, vocab_size, device)
    history_rows: list[dict[str, object]] = []

    if checkpoint_path.exists() and not args.force_retrain:
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        config.update(payload.get("config", {}))
        history_rows.append({"epoch": 0, "train_loss": None, "source": "cached"})
    else:
        train_dataset = LastEventWindowDataset(train_prepared.user_sequences, max_len=args.history_len)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        history = train_model(model, train_loader, device=device, epochs=args.epochs, lr=args.lr)
        for epoch, loss_value in enumerate(history, start=1):
            history_rows.append({"epoch": epoch, "train_loss": loss_value, "source": "train"})
        torch.save({"config": config, "model_state_dict": model.state_dict()}, checkpoint_path)

    metric_rows = []
    for split_name, prepared in [("val", val_prepared), ("test", test_prepared)]:
        examples = build_last_event_examples(prepared.user_sequences, max_len=args.history_len)
        metrics = evaluate_next_event_model_on_examples(model, examples, device=device, batch_size=2048)
        metric_rows.append({"split": split_name, **metrics})

    return model, config, pl.DataFrame(history_rows), pl.DataFrame(metric_rows)


def summarize_metrics(metrics: pl.DataFrame) -> pl.DataFrame:
    return (
        metrics.group_by(["label", "feature_set", "split"], maintain_order=True)
        .agg(
            [
                pl.col("roc_auc").mean().alias("roc_auc_mean"),
                pl.col("roc_auc").std().fill_null(0.0).alias("roc_auc_std"),
                pl.col("pr_auc").mean().alias("pr_auc_mean"),
                pl.col("pr_auc").std().fill_null(0.0).alias("pr_auc_std"),
                pl.col("num_users").first().alias("num_users"),
            ]
        )
        .sort(["label", "split", "feature_set"])
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.out_dir / "checkpoints"
    embeddings_dir = args.out_dir / "embeddings"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(42)
    prepared_by_split = load_split_sequences(args.simclr_dir, max_users_per_split=args.max_users_per_split)
    train_prepared = prepared_by_split["train"]
    val_prepared = prepared_by_split["val"]
    test_prepared = prepared_by_split["test"]
    prepared = combine_prepared(prepared_by_split)
    split_frame = build_split_frame({split: prepared_by_split[split].user_ids for split in ["train", "val", "test"]})
    split_frame.write_csv(args.out_dir / "split_users.csv")

    vocab_size = len(prepared.event2idx)
    device = choose_device()
    checkpoint_path = checkpoint_dir / f"simclr_aligned_gru_h{args.hidden_dim}_l{args.num_layers}_last{args.history_len}.pt"
    model, model_config, training_history, next_event_metrics = train_or_load_model(
        args,
        train_prepared,
        val_prepared,
        test_prepared,
        vocab_size,
        checkpoint_path,
        device,
    )
    training_history.write_csv(args.out_dir / "aligned_training_history.csv")
    next_event_metrics.write_csv(args.out_dir / "aligned_next_event_metrics.csv")

    embeddings = build_user_embeddings_with_pooling(
        model,
        prepared,
        max_len=args.history_len,
        batch_size=args.embed_batch_size,
        device=device,
        pooling=args.pooling,
        take_last=True,
    )
    embeddings_path = embeddings_dir / f"rnn_{args.pooling}_last{args.history_len}_simclr_aligned.parquet"
    embeddings.write_parquet(embeddings_path)

    labels_with_split = load_labels(args.simclr_dir, split_frame)
    label_cols = [col for col in labels_with_split.columns if col.startswith("retention_")]
    label_summary = labels_with_split.group_by("split").agg(
        [pl.len().alias("users"), *[pl.col(label).mean().alias(f"{label}_positive_rate") for label in label_cols]]
    )
    label_summary.write_csv(args.out_dir / "aligned_label_summary.csv")

    if args.skip_eval:
        metrics_df = pl.DataFrame()
        summary_df = pl.DataFrame()
    else:
        baseline_df, baseline_cols = load_baseline_features(args.simclr_dir, labels_with_split)
        emb_cols = [col for col in embeddings.columns if col.startswith("emb_")]
        eval_df = baseline_df.join(embeddings, on="appmetrica_device_id", how="inner")
        metric_frames: list[pl.DataFrame] = []
        for seed in args.seeds:
            for label in label_cols:
                for feature_set, cols in [
                    ("baseline", baseline_cols),
                    ("rnn_embedding", emb_cols),
                    ("baseline_plus_rnn_embedding", [*baseline_cols, *emb_cols]),
                ]:
                    metrics = evaluate_xgboost_feature_set(eval_df, cols, label, feature_set, seed=seed)
                    metric_frames.append(metrics.with_columns(pl.lit(seed).alias("seed")))
        metrics_df = pl.concat(metric_frames, how="vertical")
        summary_df = summarize_metrics(metrics_df)

    if metrics_df.height:
        metrics_df.write_csv(args.out_dir / "aligned_metrics.csv")
    if summary_df.height:
        summary_df.write_csv(args.out_dir / "aligned_summary.csv")

    data_summary = pl.DataFrame(
        {
            "split": ["train", "val", "test", "all"],
            "users": [len(train_prepared.user_ids), len(val_prepared.user_ids), len(test_prepared.user_ids), len(prepared.user_ids)],
            "median_nonpad_len": [
                float(np.median([len(seq) for seq in train_prepared.user_sequences])),
                float(np.median([len(seq) for seq in val_prepared.user_sequences])),
                float(np.median([len(seq) for seq in test_prepared.user_sequences])),
                float(np.median([len(seq) for seq in prepared.user_sequences])),
            ],
        }
    )
    data_summary.write_csv(args.out_dir / "aligned_data_summary.csv")

    manifest = {
        "task": "rnn_full_history_simclr_aligned",
        "simclr_dir": str(args.simclr_dir),
        "history_mode": f"last_{args.history_len}_events_from_simclr_processed_sequences",
        "embedding_file": str(embeddings_path),
        "metrics_file": None if args.skip_eval else str(args.out_dir / "aligned_metrics.csv"),
        "summary_file": None if args.skip_eval else str(args.out_dir / "aligned_summary.csv"),
        "label_summary_file": str(args.out_dir / "aligned_label_summary.csv"),
        "data_summary_file": str(args.out_dir / "aligned_data_summary.csv"),
        "training_history_file": str(args.out_dir / "aligned_training_history.csv"),
        "next_event_metrics_file": str(args.out_dir / "aligned_next_event_metrics.csv"),
        "checkpoint": str(checkpoint_path),
        "pooling": args.pooling,
        "history_len": args.history_len,
        "seeds": args.seeds,
        "model_config": model_config,
        "artifact_contract": {
            "join_key": "appmetrica_device_id",
            "source_split": "SimCLR processed user_ids_train/val/test.npy",
            "labels": label_cols,
            "split_column": "split",
            "label_definition": "SimCLR retention_labels.csv duration labels",
            "coverage": {
                "train": len(train_prepared.user_ids),
                "val": len(val_prepared.user_ids),
                "test": len(test_prepared.user_ids),
            },
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
