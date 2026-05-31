from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    EventWindowDataset,
    LastEventWindowDataset,
    PreparedSequences,
    build_last_event_examples,
    build_next_event_examples,
    load_prepared_sequences,
)
from src.downstream import (
    build_base_meta,
    build_full_history_baseline_features,
    build_meta_for_full_history,
    evaluate_xgboost_feature_set,
)
from src.eval import evaluate_next_event_model_on_examples
from src.models import GRUEncoder, RecurrentEncoder
from src.pooling import build_user_embeddings_with_pooling
from src.train import choose_device, seed_everything, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and evaluate RNN full-history user embeddings.")
    parser.add_argument("--prepared-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "data")
    parser.add_argument(
        "--clean-events-glob",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "data" / "clean_events" / "*.parquet"),
    )
    parser.add_argument("--split-csv", type=Path, default=PROJECT_ROOT / "artifacts" / "data" / "master_split_lesha.csv")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "rnn_full_history")
    parser.add_argument("--history-len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--train-window-mode", choices=["last", "rolling"], default="last")
    parser.add_argument("--event-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--pooling", choices=["last", "mean", "max", "last_mean"], default="max")
    parser.add_argument("--embed-batch-size", type=int, default=512)
    parser.add_argument("--top-k-events", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 77, 123, 202])
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument(
        "--max-users-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit. Full coursework runs should leave this unset.",
    )
    return parser.parse_args()


def subset_prepared(prepared: PreparedSequences, user_ids: set[str]) -> PreparedSequences:
    kept_user_ids: list[str] = []
    kept_sequences: list[list[int]] = []
    for user_id, seq in zip(prepared.user_ids, prepared.user_sequences, strict=True):
        if user_id in user_ids:
            kept_user_ids.append(user_id)
            kept_sequences.append(seq)
    return PreparedSequences(user_ids=kept_user_ids, user_sequences=kept_sequences, event2idx=prepared.event2idx)


def load_model(checkpoint_path: Path, vocab_size: int, device: torch.device) -> tuple[RecurrentEncoder, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint["config"])
    state_dict = checkpoint["model_state_dict"]
    event_dim = int(state_dict["embedding.weight"].shape[1])
    hidden_dim = int(state_dict["projection.weight"].shape[1])

    model = RecurrentEncoder(
        vocab_size=vocab_size,
        event_dim=event_dim,
        hidden_dim=hidden_dim,
        num_layers=int(config.get("num_layers", 1)),
        dropout=float(config.get("dropout", 0.0)),
        rnn_type=str(config.get("model_type", "gru")),
    )
    model.load_state_dict(state_dict)
    model.to(device)

    config.update({"event_dim": event_dim, "hidden_dim": hidden_dim, "vocab_size": vocab_size})
    return model, config


def make_full_history_model(args: argparse.Namespace, vocab_size: int, device: torch.device) -> RecurrentEncoder:
    return GRUEncoder(
        vocab_size=vocab_size,
        event_dim=args.event_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)


def train_or_load_full_history_model(
    args: argparse.Namespace,
    train_prepared: PreparedSequences,
    val_prepared: PreparedSequences,
    test_prepared: PreparedSequences,
    vocab_size: int,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RecurrentEncoder, dict[str, object], pl.DataFrame, pl.DataFrame]:
    config = {
        "experiment_id": f"gru_h{args.hidden_dim}_l{args.num_layers}_last{args.history_len}",
        "model_type": "gru",
        "event_dim": args.event_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "max_len": args.history_len,
        "stride": args.stride,
        "train_window_mode": args.train_window_mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_batch_size": args.train_batch_size,
        "vocab_size": vocab_size,
        "seed": 42,
    }

    if args.train_window_mode == "last":
        train_dataset = LastEventWindowDataset(train_prepared.user_sequences, max_len=args.history_len)
        val_examples = build_last_event_examples(val_prepared.user_sequences, max_len=args.history_len)
        test_examples = build_last_event_examples(test_prepared.user_sequences, max_len=args.history_len)
    else:
        train_dataset = EventWindowDataset(train_prepared.user_sequences, max_len=args.history_len, stride=args.stride)
        val_examples = build_next_event_examples(val_prepared.user_sequences, max_len=args.history_len, stride=args.stride)
        test_examples = build_next_event_examples(test_prepared.user_sequences, max_len=args.history_len, stride=args.stride)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    model = make_full_history_model(args, vocab_size, device)
    history_rows: list[dict[str, object]] = []

    if checkpoint_path.exists() and not args.force_retrain:
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        config.update(payload.get("config", {}))
        history_rows.append({"epoch": 0, "train_loss": None, "source": "cached"})
    else:
        history = train_model(model, train_loader, device=device, epochs=args.epochs, lr=args.lr)
        for epoch, loss_value in enumerate(history, start=1):
            history_rows.append({"epoch": epoch, "train_loss": loss_value, "source": "train"})
        torch.save({"config": config, "model_state_dict": model.state_dict()}, checkpoint_path)

    metric_rows = []
    for split_name, examples in [("val", val_examples), ("test", test_examples)]:
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
    prepared = load_prepared_sequences(args.prepared_dir)
    split_users = pl.read_csv(args.split_csv, schema_overrides={"appmetrica_device_id": pl.String})
    if args.max_users_per_split is not None:
        split_users = pl.concat(
            [
                split_users.filter(pl.col("split") == split_name).head(args.max_users_per_split)
                for split_name in ["train", "val", "test"]
            ],
            how="vertical",
        )
    split_user_ids = set(split_users["appmetrica_device_id"].to_list())
    prepared = subset_prepared(prepared, split_user_ids)

    train_ids = set(split_users.filter(pl.col("split") == "train")["appmetrica_device_id"].to_list())
    val_ids = set(split_users.filter(pl.col("split") == "val")["appmetrica_device_id"].to_list())
    test_ids = set(split_users.filter(pl.col("split") == "test")["appmetrica_device_id"].to_list())
    train_prepared = subset_prepared(prepared, train_ids)
    val_prepared = subset_prepared(prepared, val_ids)
    test_prepared = subset_prepared(prepared, test_ids)

    device = choose_device()
    checkpoint_path = checkpoint_dir / f"gru_h{args.hidden_dim}_l{args.num_layers}_last{args.history_len}.pt"
    model, model_config, training_history, next_event_metrics = train_or_load_full_history_model(
        args,
        train_prepared,
        val_prepared,
        test_prepared,
        vocab_size=len(prepared.event2idx),
        checkpoint_path=checkpoint_path,
        device=device,
    )
    training_history.write_csv(args.out_dir / "full_history_training_history.csv")
    next_event_metrics.write_csv(args.out_dir / "full_history_next_event_metrics.csv")

    embeddings = build_user_embeddings_with_pooling(
        model,
        prepared,
        max_len=args.history_len,
        batch_size=args.embed_batch_size,
        device=device,
        pooling=args.pooling,
        take_last=True,
    )
    embeddings_path = embeddings_dir / f"rnn_{args.pooling}_last{args.history_len}_full_history.parquet"
    embeddings.write_parquet(embeddings_path)

    horizons = {"retention_7d": 7, "retention_14d": 14, "retention_30d": 30}
    base_meta, data_end_ts = build_base_meta(args.clean_events_glob, prepared.user_ids)
    meta, label_summary = build_meta_for_full_history(base_meta, horizons=horizons)
    label_summary.write_csv(args.out_dir / "full_history_label_summary.csv")

    baseline_df, baseline_cols, baseline_summary = build_full_history_baseline_features(
        prepared,
        train_prepared,
        split_users,
        meta,
        max_history_len=args.history_len,
        top_k_events=args.top_k_events,
        horizons=horizons,
    )
    baseline_summary.write_csv(args.out_dir / "full_history_baseline_summary.csv")

    emb_cols = [col for col in embeddings.columns if col.startswith("emb_")]
    eval_df = baseline_df.join(embeddings, on="appmetrica_device_id", how="inner")

    metric_frames: list[pl.DataFrame] = []
    for seed in args.seeds:
        for label in horizons:
            for feature_set, cols in [
                ("baseline", baseline_cols),
                ("rnn_embedding", emb_cols),
                ("baseline_plus_rnn_embedding", [*baseline_cols, *emb_cols]),
            ]:
                metrics = evaluate_xgboost_feature_set(eval_df, cols, label, feature_set, seed=seed)
                metric_frames.append(metrics.with_columns(pl.lit(seed).alias("seed")))

    metrics_df = pl.concat(metric_frames, how="vertical")
    metrics_df.write_csv(args.out_dir / "full_history_metrics.csv")
    summary_df = summarize_metrics(metrics_df)
    summary_df.write_csv(args.out_dir / "full_history_summary.csv")

    manifest = {
        "task": "rnn_full_history_user_embeddings",
        "history_mode": f"last_{args.history_len}_events_from_full_history",
        "embedding_file": str(embeddings_path),
        "metrics_file": str(args.out_dir / "full_history_metrics.csv"),
        "summary_file": str(args.out_dir / "full_history_summary.csv"),
        "label_summary_file": str(args.out_dir / "full_history_label_summary.csv"),
        "training_history_file": str(args.out_dir / "full_history_training_history.csv"),
        "next_event_metrics_file": str(args.out_dir / "full_history_next_event_metrics.csv"),
        "checkpoint": str(checkpoint_path),
        "split_csv": str(args.split_csv),
        "clean_events_glob": args.clean_events_glob,
        "prepared_users": len(prepared.user_ids),
        "data_end_ts": data_end_ts,
        "pooling": args.pooling,
        "history_len": args.history_len,
        "horizons": horizons,
        "seeds": args.seeds,
        "model_config": model_config,
        "artifact_contract": {
            "join_key": "appmetrica_device_id",
            "embedding_columns": "emb_000..emb_N",
            "labels": list(horizons.keys()),
            "split_column": "split",
            "label_definition": "duration label: last_event_ts - first_event_ts >= horizon_days",
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
