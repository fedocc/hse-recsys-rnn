from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_rnn_experiments


class RunRnnExperimentsTest(unittest.TestCase):
    def test_select_series_expands_all_in_config_order(self) -> None:
        config = {
            "series": {
                "context_saturation": {},
                "pooling_ablation": {},
                "capacity_sweep": {},
            }
        }

        selected = run_rnn_experiments.select_series(config, "all")

        self.assertEqual(selected, ["context_saturation", "pooling_ablation", "capacity_sweep"])

    def test_build_artifact_plan_uses_known_sources_for_selected_series(self) -> None:
        plan = run_rnn_experiments.build_artifact_plan(["pooling_ablation"])

        source_paths = [item.source for item in plan]

        self.assertIn(Path("artifacts/pooling_ablation/pooling_downstream_metrics.csv"), source_paths)
        self.assertIn(Path("artifacts/pooling_ablation/figures/pooling_roc_auc.png"), source_paths)
        self.assertTrue(all(item.series == "pooling_ablation" for item in plan))

    def test_materialize_plan_copies_existing_artifacts_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project_root = root / "project"
            source_dir = project_root / "artifacts" / "pooling_ablation" / "figures"
            source_dir.mkdir(parents=True)
            metrics_path = project_root / "artifacts" / "pooling_ablation" / "pooling_downstream_metrics.csv"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text("split,roc_auc\ntrain,0.7\n", encoding="utf-8")
            figure_path = source_dir / "pooling_roc_auc.png"
            figure_path.write_bytes(b"png")

            out_dir = root / "out"
            paths = run_rnn_experiments.ensure_output_tree(out_dir)
            plan = [
                run_rnn_experiments.ArtifactSpec(
                    series="pooling_ablation",
                    kind="table",
                    source=Path("artifacts/pooling_ablation/pooling_downstream_metrics.csv"),
                ),
                run_rnn_experiments.ArtifactSpec(
                    series="pooling_ablation",
                    kind="figure",
                    source=Path("artifacts/pooling_ablation/figures/pooling_roc_auc.png"),
                ),
            ]

            rows = run_rnn_experiments.materialize_plan(
                plan,
                project_root=project_root,
                paths=paths,
                strict=True,
            )
            run_rnn_experiments.write_runner_outputs(
                out_dir=out_dir,
                config_path=project_root / "configs" / "rnn_experiments.yaml",
                selected_series=["pooling_ablation"],
                summary_rows=rows,
                dry_run=False,
            )

            copied_metrics = out_dir / "tables" / "pooling_ablation__pooling_downstream_metrics.csv"
            copied_figure = out_dir / "figures" / "pooling_ablation__pooling_roc_auc.png"
            summary_path = out_dir / "runner_summary.csv"
            manifest_path = out_dir / "runner_manifest.json"

            self.assertTrue(copied_metrics.exists())
            self.assertTrue(copied_figure.exists())
            with summary_path.open(newline="", encoding="utf-8") as file:
                summary_rows = list(csv.DictReader(file))
            self.assertEqual(len(summary_rows), 2)
            self.assertEqual(summary_rows[0]["rows"], "1")
            self.assertEqual(summary_rows[0]["sha256"], run_rnn_experiments.sha256_file(metrics_path))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_series"], ["pooling_ablation"])
            self.assertEqual(manifest["status"], "ok")


if __name__ == "__main__":
    unittest.main()
