from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUT_PATH = ARTIFACTS_DIR / "results_summary.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("-inf")
    return float(value)


def add_row(
    rows: list[dict[str, str]],
    *,
    section: str,
    metric: str,
    value: float | int | str,
    details: str,
) -> None:
    if isinstance(value, float):
        value_text = f"{value:.6f}"
    else:
        value_text = str(value)
    rows.append(
        {
            "section": section,
            "metric": metric,
            "value": value_text,
            "details": details,
        }
    )


def summarize_sequence_modeling(rows: list[dict[str, str]]) -> None:
    metrics = read_csv(ARTIFACTS_DIR / "rnn" / "comparison_metrics.csv")
    test_metrics = [row for row in metrics if row.get("split") == "test"]
    for row in sorted(test_metrics, key=lambda item: item.get("model", "")):
        model = row["model"]
        add_row(rows, section="sequence_modeling", metric=f"{model}_mrr_test", value=float(row["mrr"]), details="next-event prediction")
        add_row(rows, section="sequence_modeling", metric=f"{model}_hit10_test", value=float(row["hit@10"]), details="next-event prediction")


def summarize_downstream_prefix(rows: list[dict[str, str]]) -> None:
    metrics = read_csv(ARTIFACTS_DIR / "downstream" / "downstream_metrics.csv")
    test_rows = [row for row in metrics if row.get("split") == "test" and row.get("experiment") == "gru_prefix_ablation"]
    for label in sorted({row["label"] for row in test_rows}):
        label_rows = [row for row in test_rows if row["label"] == label]
        for metric_name in ["roc_auc", "pr_auc"]:
            best = max(label_rows, key=lambda item: as_float(item, metric_name))
            add_row(
                rows,
                section="downstream_prefix_ablation",
                metric=f"best_{label}_{metric_name}_test",
                value=float(best[metric_name]),
                details=f"prefix_len={best['prefix_len']}; feature_set={best['feature_set']}; num_users={best['num_users']}",
            )


def summarize_pooling(rows: list[dict[str, str]]) -> None:
    metrics = read_csv(ARTIFACTS_DIR / "pooling_ablation" / "pooling_downstream_metrics.csv")
    test_rows = [row for row in metrics if row.get("split") == "test"]
    for label in sorted({row["label"] for row in test_rows}):
        label_rows = [row for row in test_rows if row["label"] == label]
        for metric_name in ["roc_auc", "pr_auc"]:
            best = max(label_rows, key=lambda item: as_float(item, metric_name))
            add_row(
                rows,
                section="pooling_ablation",
                metric=f"best_{label}_{metric_name}_test",
                value=float(best[metric_name]),
                details=(
                    f"pooling={best['pooling']}; feature_set={best['feature_set']}; "
                    f"num_features={best['num_features']}; num_users={best['num_users']}"
                ),
            )


def summarize_supervised(rows: list[dict[str, str]]) -> None:
    metrics = read_csv(ARTIFACTS_DIR / "supervised_finetuning" / "supervised_vs_reference_test_metrics.csv")
    test_rows = [row for row in metrics if row.get("split") == "test"]
    for label in sorted({row["label"] for row in test_rows}):
        label_rows = [row for row in test_rows if row["label"] == label]
        for metric_name in ["roc_auc", "pr_auc"]:
            best = max(label_rows, key=lambda item: as_float(item, metric_name))
            add_row(
                rows,
                section="supervised_vs_reference",
                metric=f"best_{label}_{metric_name}_test",
                value=float(best[metric_name]),
                details=f"mode={best['mode']}; num_users={best['num_users']}",
            )


def summarize_capacity_sweep(rows: list[dict[str, str]]) -> None:
    next_event = read_csv(ARTIFACTS_DIR / "capacity_sweep" / "capacity_next_event_metrics.csv")
    test_next_event = [row for row in next_event if row.get("split") == "test"]
    if test_next_event:
        for metric_name in ["mrr", "hit@5", "hit@10"]:
            best = max(test_next_event, key=lambda item: as_float(item, metric_name))
            add_row(
                rows,
                section="capacity_sweep",
                metric=f"best_next_event_{metric_name.replace('@', '')}_test",
                value=float(best[metric_name]),
                details=(
                    f"experiment_id={best['experiment_id']}; model_type={best['model_type']}; "
                    f"hidden_dim={best['hidden_dim']}; num_layers={best['num_layers']}"
                ),
            )

    downstream = read_csv(ARTIFACTS_DIR / "capacity_sweep" / "capacity_downstream_metrics.csv")
    test_downstream = [row for row in downstream if row.get("split") == "test"]
    if not test_downstream:
        return

    for label in sorted({row["label"] for row in test_downstream}):
        label_rows = [row for row in test_downstream if row["label"] == label]
        for feature_group in sorted({row["feature_group"] for row in label_rows}):
            group_rows = [row for row in label_rows if row["feature_group"] == feature_group]
            for metric_name in ["roc_auc", "pr_auc"]:
                best = max(group_rows, key=lambda item: as_float(item, metric_name))
                add_row(
                    rows,
                    section="capacity_sweep",
                    metric=f"best_{label}_{feature_group}_{metric_name}_test",
                    value=float(best[metric_name]),
                    details=(
                        f"experiment_id={best['experiment_id']}; model_type={best['model_type']}; "
                        f"hidden_dim={best['hidden_dim']}; num_layers={best['num_layers']}; "
                        f"pooling={best['pooling']}; num_users={best['num_users']}"
                    ),
                )


def summarize_final_stability(rows: list[dict[str, str]]) -> None:
    metrics = read_csv(ARTIFACTS_DIR / "final_stability" / "stability_summary.csv")
    if not metrics:
        return

    for label in sorted({row["label"] for row in metrics}):
        label_rows = [row for row in metrics if row["label"] == label]
        for feature_group in sorted({row["feature_group"] for row in label_rows}):
            group_rows = [row for row in label_rows if row["feature_group"] == feature_group]
            best = max(group_rows, key=lambda item: as_float(item, "roc_auc_mean"))
            add_row(
                rows,
                section="final_stability",
                metric=f"best_{label}_{feature_group}_roc_auc_mean_test",
                value=float(best["roc_auc_mean"]),
                details=(
                    f"candidate={best['candidate']}; roc_auc_std={float(best['roc_auc_std']):.6f}; "
                    f"pr_auc_mean={float(best['pr_auc_mean']):.6f}; pr_auc_std={float(best['pr_auc_std']):.6f}; "
                    f"num_users={best['num_users']}"
                ),
            )


def summarize_full_history_aligned(rows: list[dict[str, str]]) -> None:
    metrics = read_csv(ARTIFACTS_DIR / "rnn_full_history_simclr_aligned" / "aligned_summary.csv")
    test_rows = [row for row in metrics if row.get("split") == "test"]
    if not test_rows:
        return

    for label in sorted({row["label"] for row in test_rows}):
        label_rows = [row for row in test_rows if row["label"] == label]
        for feature_set in sorted({row["feature_set"] for row in label_rows}):
            row = [item for item in label_rows if item["feature_set"] == feature_set][0]
            slug = feature_set.lower().replace(" + ", "_plus_").replace("-", "_").replace(" ", "_")
            add_row(
                rows,
                section="full_history_aligned",
                metric=f"{label}_{slug}_roc_auc_mean_test",
                value=float(row["roc_auc_mean"]),
                details=(
                    f"roc_auc_std={float(row['roc_auc_std']):.6f}; "
                    f"pr_auc_mean={float(row['pr_auc_mean']):.6f}; "
                    f"pr_auc_std={float(row['pr_auc_std']):.6f}; "
                    f"num_users={row['num_users']}"
                ),
            )


def summarize_negative_controls(rows: list[dict[str, str]]) -> None:
    metrics = read_csv(ARTIFACTS_DIR / "negative_controls" / "negative_control_metrics.csv")
    test_rows = [row for row in metrics if row.get("split") == "test"]
    if not test_rows:
        return

    for label in sorted({row["label"] for row in test_rows}):
        label_rows = [row for row in test_rows if row["label"] == label]
        for feature_set in sorted({row["feature_set"] for row in label_rows}):
            feature_rows = [row for row in label_rows if row["feature_set"] == feature_set]
            if not feature_rows:
                continue
            row = feature_rows[0]
            slug = feature_set.lower().replace(" + ", "_plus_").replace("-", "_").replace(" ", "_")
            add_row(
                rows,
                section="negative_controls",
                metric=f"{label}_{slug}_roc_auc_test",
                value=float(row["roc_auc"]),
                details=f"feature_set={feature_set}; num_users={row['num_users']}",
            )
            add_row(
                rows,
                section="negative_controls",
                metric=f"{label}_{slug}_pr_auc_test",
                value=float(row["pr_auc"]),
                details=f"feature_set={feature_set}; num_users={row['num_users']}",
            )


def summarize_common_valid_prefix(rows: list[dict[str, str]]) -> None:
    metrics = read_csv(ARTIFACTS_DIR / "common_valid_prefix_ablation" / "common_valid_best_test_rows.csv")
    if not metrics:
        return

    for row in metrics:
        slug = row["feature_set"].lower().replace(" + ", "_plus_").replace("-", "_").replace(" ", "_")
        add_row(
            rows,
            section="common_valid_prefix_ablation",
            metric=f"best_{row['label']}_{slug}_roc_auc_test",
            value=float(row["roc_auc"]),
            details=f"prefix_len={row['prefix_len']}; feature_set={row['feature_set']}; num_users={row['num_users']}",
        )
        add_row(
            rows,
            section="common_valid_prefix_ablation",
            metric=f"best_{row['label']}_{slug}_pr_auc_test",
            value=float(row["pr_auc"]),
            details=f"prefix_len={row['prefix_len']}; feature_set={row['feature_set']}; num_users={row['num_users']}",
        )


def summarize_order_sensitivity(rows: list[dict[str, str]]) -> None:
    summary = read_csv(ARTIFACTS_DIR / "order_sensitivity" / "order_sensitivity_summary.csv")
    if summary:
        row = summary[0]
        for metric_name in [
            "mean_cosine",
            "median_cosine",
            "p05_cosine",
            "p95_cosine",
            "mean_relative_l2",
            "median_relative_l2",
        ]:
            add_row(
                rows,
                section="order_sensitivity",
                metric=metric_name,
                value=float(row[metric_name]),
                details="GRU mean vs shuffled-prefix GRU mean",
            )

    correlations = read_csv(ARTIFACTS_DIR / "order_sensitivity" / "order_sensitivity_correlations.csv")
    for row in correlations:
        add_row(
            rows,
            section="order_sensitivity_correlations",
            metric=f"{row['feature']}_corr_with_relative_l2",
            value=float(row["corr_with_relative_l2"]),
            details="correlation with order_relative_l2_distance",
        )


def summarize_embedding_space(rows: list[dict[str, str]]) -> None:
    metrics = read_csv(ARTIFACTS_DIR / "embedding_analysis" / "embedding_space_summary.csv")
    for row in sorted(metrics, key=lambda item: item.get("model", "")):
        model = row["model"]
        add_row(rows, section="embedding_space", metric=f"{model}_mean_norm", value=float(row["mean_norm"]), details="full-history embeddings")
        add_row(
            rows,
            section="embedding_space",
            metric=f"{model}_pca10_cumulative_variance",
            value=float(row["pca10_cumulative_variance"]),
            details="full-history embeddings",
        )


def main() -> None:
    rows: list[dict[str, str]] = []
    summarize_sequence_modeling(rows)
    summarize_downstream_prefix(rows)
    summarize_pooling(rows)
    summarize_supervised(rows)
    summarize_capacity_sweep(rows)
    summarize_final_stability(rows)
    summarize_full_history_aligned(rows)
    summarize_negative_controls(rows)
    summarize_common_valid_prefix(rows)
    summarize_order_sensitivity(rows)
    summarize_embedding_space(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["section", "metric", "value", "details"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
