from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rnn_experiments.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "artifacts" / "experiments" / "rnn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RNN experiment runner scaffold.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--series",
        choices=["context_saturation", "pooling_ablation", "capacity_sweep", "supervised_retention", "all"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print the planned experiments.")
    return parser.parse_args()


def ensure_output_tree(out_dir: Path) -> dict[str, Path]:
    paths = {
        "root": out_dir,
        "configs": out_dir / "configs",
        "checkpoints": out_dir / "checkpoints",
        "embeddings": out_dir / "embeddings",
        "figures": out_dir / "figures",
        "tables": out_dir / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def main() -> None:
    args = parse_args()
    paths = ensure_output_tree(args.out_dir)

    status = {
        "config": str(args.config),
        "series": args.series,
        "dry_run": args.dry_run,
        "out_dir": str(args.out_dir),
        "created_dirs": {key: str(value) for key, value in paths.items()},
        "next_step": "Implement concrete series execution: context_saturation -> pooling_ablation -> capacity_sweep.",
    }

    status_path = args.out_dir / "runner_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
