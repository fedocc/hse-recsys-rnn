from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rnn_experiments.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "rnn"


@dataclass(frozen=True)
class ArtifactSpec:
    series: str
    kind: str
    source: Path
    copy: bool = True


ARTIFACTS_BY_SERIES: dict[str, list[ArtifactSpec]] = {
    "context_saturation": [
        ArtifactSpec("context_saturation", "table", Path("artifacts/downstream/downstream_metrics.csv")),
        ArtifactSpec("context_saturation", "table", Path("artifacts/common_valid_prefix_ablation/common_valid_prefix_metrics.csv")),
        ArtifactSpec("context_saturation", "table", Path("artifacts/common_valid_prefix_ablation/common_valid_best_test_rows.csv")),
        ArtifactSpec("context_saturation", "table", Path("artifacts/common_valid_prefix_ablation/label_summary_by_prefix.csv")),
        ArtifactSpec("context_saturation", "figure", Path("artifacts/common_valid_prefix_ablation/figures/common_valid_prefix_metrics.png")),
    ],
    "pooling_ablation": [
        ArtifactSpec("pooling_ablation", "table", Path("artifacts/pooling_ablation/pooling_downstream_metrics.csv")),
        ArtifactSpec("pooling_ablation", "table", Path("artifacts/pooling_ablation/pooling_embedding_summary.csv")),
        ArtifactSpec("pooling_ablation", "table", Path("artifacts/pooling_ablation/pooling_space_summary.csv")),
        ArtifactSpec("pooling_ablation", "figure", Path("artifacts/pooling_ablation/figures/pooling_roc_auc.png")),
        ArtifactSpec("pooling_ablation", "figure", Path("artifacts/pooling_ablation/figures/pooling_pr_auc.png")),
        ArtifactSpec("pooling_ablation", "figure", Path("artifacts/pooling_ablation/figures/pooling_space_diagnostics.png")),
    ],
    "capacity_sweep": [
        ArtifactSpec("capacity_sweep", "table", Path("artifacts/capacity_sweep/capacity_config.csv")),
        ArtifactSpec("capacity_sweep", "table", Path("artifacts/capacity_sweep/capacity_next_event_metrics.csv")),
        ArtifactSpec("capacity_sweep", "table", Path("artifacts/capacity_sweep/capacity_downstream_metrics.csv")),
        ArtifactSpec("capacity_sweep", "table", Path("artifacts/capacity_sweep/capacity_final_table.csv")),
        ArtifactSpec("capacity_sweep", "table", Path("artifacts/capacity_sweep/capacity_embedding_summary.csv")),
        ArtifactSpec("capacity_sweep", "figure", Path("artifacts/capacity_sweep/figures/capacity_next_event_metrics.png")),
        ArtifactSpec("capacity_sweep", "figure", Path("artifacts/capacity_sweep/figures/capacity_downstream_metrics.png")),
        ArtifactSpec("capacity_sweep", "checkpoint", Path("artifacts/capacity_sweep/checkpoints/gru_h256_l1.pt"), copy=False),
    ],
    "supervised_retention": [
        ArtifactSpec("supervised_retention", "table", Path("artifacts/supervised_finetuning/supervised_metrics.csv")),
        ArtifactSpec("supervised_retention", "table", Path("artifacts/supervised_finetuning/supervised_vs_reference_test_metrics.csv")),
        ArtifactSpec("supervised_retention", "table", Path("artifacts/supervised_finetuning/supervised_artifacts.csv")),
        ArtifactSpec("supervised_retention", "figure", Path("artifacts/supervised_finetuning/figures/supervised_roc_auc_comparison.png")),
        ArtifactSpec("supervised_retention", "figure", Path("artifacts/supervised_finetuning/figures/supervised_pr_auc_comparison.png")),
    ],
    "final_stability": [
        ArtifactSpec("final_stability", "table", Path("artifacts/final_stability/stability_summary.csv")),
        ArtifactSpec("final_stability", "table", Path("artifacts/final_stability/final_candidate_summary.csv")),
        ArtifactSpec("final_stability", "table", Path("artifacts/final_stability/stability_best_by_group.csv")),
        ArtifactSpec("final_stability", "figure", Path("artifacts/final_stability/figures/stability_metrics.png")),
        ArtifactSpec("final_stability", "figure", Path("artifacts/final_stability/figures/stability_seed_lines.png")),
    ],
    "negative_controls": [
        ArtifactSpec("negative_controls", "table", Path("artifacts/negative_controls/negative_control_metrics.csv")),
        ArtifactSpec("negative_controls", "table", Path("artifacts/negative_controls/negative_control_representations.csv")),
        ArtifactSpec("negative_controls", "figure", Path("artifacts/negative_controls/figures/negative_controls_test_metrics.png")),
    ],
    "order_sensitivity": [
        ArtifactSpec("order_sensitivity", "table", Path("artifacts/order_sensitivity/order_sensitivity_summary.csv")),
        ArtifactSpec("order_sensitivity", "table", Path("artifacts/order_sensitivity/order_sensitivity_correlations.csv")),
        ArtifactSpec("order_sensitivity", "table", Path("artifacts/order_sensitivity/order_sensitivity_bucket_profile.csv")),
        ArtifactSpec("order_sensitivity", "figure", Path("artifacts/order_sensitivity/figures/order_sensitivity_diagnostics.png")),
    ],
    "embedding_analysis": [
        ArtifactSpec("embedding_analysis", "table", Path("artifacts/embedding_analysis/embedding_space_summary.csv")),
        ArtifactSpec("embedding_analysis", "table", Path("artifacts/embedding_analysis/kmeans_scores.csv")),
        ArtifactSpec("embedding_analysis", "table", Path("artifacts/embedding_analysis/gru_kmeans_profile.csv")),
        ArtifactSpec("embedding_analysis", "figure", Path("artifacts/embedding_analysis/figures/pca_cumulative_variance.png")),
        ArtifactSpec("embedding_analysis", "figure", Path("artifacts/embedding_analysis/figures/gru_clusters_pca.png")),
        ArtifactSpec("embedding_analysis", "figure", Path("artifacts/embedding_analysis/figures/gru_cluster_profiles.png")),
    ],
    "full_history": [
        ArtifactSpec("full_history", "table", Path("artifacts/rnn_full_history/full_history_summary.csv")),
        ArtifactSpec("full_history", "table", Path("artifacts/rnn_full_history/full_history_next_event_metrics.csv")),
        ArtifactSpec("full_history", "table", Path("artifacts/rnn_full_history/full_history_metrics.csv")),
        ArtifactSpec("full_history", "table", Path("artifacts/rnn_full_history/full_history_embedding_summary.csv")),
        ArtifactSpec("full_history", "manifest", Path("artifacts/rnn_full_history/manifest.json")),
        ArtifactSpec("full_history", "figure", Path("artifacts/rnn_full_history/figures/full_history_downstream_metrics.png")),
    ],
    "full_history_aligned": [
        ArtifactSpec("full_history_aligned", "table", Path("artifacts/rnn_full_history_simclr_aligned/aligned_data_summary.csv")),
        ArtifactSpec("full_history_aligned", "table", Path("artifacts/rnn_full_history_simclr_aligned/aligned_label_summary.csv")),
        ArtifactSpec("full_history_aligned", "table", Path("artifacts/rnn_full_history_simclr_aligned/aligned_next_event_metrics.csv")),
        ArtifactSpec("full_history_aligned", "table", Path("artifacts/rnn_full_history_simclr_aligned/aligned_training_history.csv")),
        ArtifactSpec("full_history_aligned", "table", Path("artifacts/rnn_full_history_simclr_aligned/aligned_metrics.csv")),
        ArtifactSpec("full_history_aligned", "table", Path("artifacts/rnn_full_history_simclr_aligned/aligned_summary.csv")),
        ArtifactSpec("full_history_aligned", "manifest", Path("artifacts/rnn_full_history_simclr_aligned/manifest.json")),
        ArtifactSpec(
            "full_history_aligned",
            "checkpoint",
            Path("artifacts/rnn_full_history_simclr_aligned/checkpoints/simclr_aligned_gru_h256_l1_last512.pt"),
            copy=False,
        ),
        ArtifactSpec(
            "full_history_aligned",
            "embedding",
            Path("artifacts/rnn_full_history_simclr_aligned/embeddings/rnn_max_last512_simclr_aligned.parquet"),
            copy=False,
        ),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and validate RNN experiment artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--series", default="all", help="Series name from config, or 'all'.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned artifacts without writing outputs.")
    parser.add_argument("--strict", action="store_true", help="Fail if an expected artifact is missing.")
    return parser.parse_args()


def ensure_output_tree(out_dir: Path) -> dict[str, Path]:
    paths = {
        "root": out_dir,
        "configs": out_dir / "configs",
        "figures": out_dir / "figures",
        "manifests": out_dir / "manifests",
        "tables": out_dir / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return {"series": {name: {} for name in parse_series_names(path)}}

    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def parse_series_names(path: Path) -> list[str]:
    names: list[str] = []
    in_series = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line:
            continue
        if line == "series:":
            in_series = True
            continue
        if in_series and not raw_line.startswith(" "):
            break
        if in_series and raw_line.startswith("  ") and not raw_line.startswith("    "):
            key = line.strip().removesuffix(":")
            if key:
                names.append(key)
    return names


def select_series(config: dict[str, Any], requested: str) -> list[str]:
    series = config.get("series", {})
    if not isinstance(series, dict) or not series:
        raise ValueError("Config must contain a non-empty 'series' mapping.")

    available = list(series)
    if requested == "all":
        return available
    if requested not in series:
        available_text = ", ".join(available)
        raise ValueError(f"Unknown series '{requested}'. Available series: {available_text}")
    return [requested]


def build_artifact_plan(selected_series: list[str]) -> list[ArtifactSpec]:
    plan: list[ArtifactSpec] = []
    for series in selected_series:
        try:
            plan.extend(ARTIFACTS_BY_SERIES[series])
        except KeyError as exc:
            raise ValueError(f"No artifact plan is registered for series '{series}'") from exc
    return plan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0
        return sum(1 for _ in reader), len(header)


def json_key_count(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return len(data)
    if isinstance(data, list):
        return len(data)
    return 1


def output_path_for(spec: ArtifactSpec, paths: dict[str, Path]) -> Path | None:
    filename = f"{spec.series}__{spec.source.name}"
    if spec.kind == "table":
        return paths["tables"] / filename
    if spec.kind == "figure":
        return paths["figures"] / filename
    if spec.kind == "manifest":
        return paths["manifests"] / filename
    return None


def materialize_plan(
    plan: list[ArtifactSpec],
    *,
    project_root: Path,
    paths: dict[str, Path],
    strict: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    missing: list[Path] = []

    for spec in plan:
        absolute_source = project_root / spec.source
        exists = absolute_source.exists()
        output_path = output_path_for(spec, paths)
        copied = False
        rows_count = ""
        columns_count = ""
        json_items = ""
        sha256 = ""
        size_bytes = ""

        if exists:
            size_bytes = str(absolute_source.stat().st_size)
            sha256 = sha256_file(absolute_source)
            if absolute_source.suffix == ".csv":
                rows_num, columns_num = csv_shape(absolute_source)
                rows_count = str(rows_num)
                columns_count = str(columns_num)
            elif absolute_source.suffix == ".json":
                json_items = str(json_key_count(absolute_source))

            if spec.copy and output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(absolute_source, output_path)
                copied = True
        else:
            missing.append(spec.source)

        rows.append(
            {
                "series": spec.series,
                "kind": spec.kind,
                "source_path": str(spec.source),
                "exists": str(exists).lower(),
                "copied": str(copied).lower(),
                "output_path": "" if output_path is None else str(output_path),
                "rows": rows_count,
                "columns": columns_count,
                "json_items": json_items,
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
        )

    if strict and missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing expected artifacts:\n{missing_text}")

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "series",
        "kind",
        "source_path",
        "exists",
        "copied",
        "output_path",
        "rows",
        "columns",
        "json_items",
        "size_bytes",
        "sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_runner_outputs(
    *,
    out_dir: Path,
    config_path: Path,
    selected_series: list[str],
    summary_rows: list[dict[str, str]],
    dry_run: bool,
) -> dict[str, Any]:
    existing_count = sum(row["exists"] == "true" for row in summary_rows)
    missing_count = sum(row["exists"] == "false" for row in summary_rows)
    status = "ok" if missing_count == 0 else "missing_artifacts"
    payload: dict[str, Any] = {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "selected_series": selected_series,
        "dry_run": dry_run,
        "artifact_count": len(summary_rows),
        "existing_artifact_count": existing_count,
        "missing_artifact_count": missing_count,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "runner_summary.csv", summary_rows)
    (out_dir / "runner_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "runner_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    config_copy_dir = out_dir / "configs"
    config_copy_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        shutil.copy2(config_path, config_copy_dir / config_path.name)

    return payload


def dry_run_payload(selected_series: list[str], plan: list[ArtifactSpec]) -> dict[str, Any]:
    return {
        "status": "dry_run",
        "selected_series": selected_series,
        "artifact_count": len(plan),
        "artifacts": [
            {
                "series": spec.series,
                "kind": spec.kind,
                "source_path": str(spec.source),
                "copy": spec.copy,
            }
            for spec in plan
        ],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    selected = select_series(config, args.series)
    plan = build_artifact_plan(selected)

    if args.dry_run:
        print(json.dumps(dry_run_payload(selected, plan), ensure_ascii=False, indent=2))
        return

    paths = ensure_output_tree(args.out_dir)
    rows = materialize_plan(plan, project_root=PROJECT_ROOT, paths=paths, strict=args.strict)
    status = write_runner_outputs(
        out_dir=args.out_dir,
        config_path=args.config,
        selected_series=selected,
        summary_rows=rows,
        dry_run=False,
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
