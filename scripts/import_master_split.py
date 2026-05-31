from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data import load_prepared_sequences
from src.splits import build_user_split_from_master_csv, save_user_split_tables


DEFAULT_MASTER_SPLIT = Path("/Users/fedornikonov/Downloads/master_split.csv")
DEFAULT_DATA_DIR = PROJECT_ROOT / "artifacts" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a team master user-level split into the RNN project.")
    parser.add_argument("--master-split", type=Path, default=DEFAULT_MASTER_SPLIT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = load_prepared_sequences(args.data_dir)
    split, summary_df = build_user_split_from_master_csv(prepared.user_ids, args.master_split, seed=args.seed)

    print(summary_df)
    if args.dry_run:
        return

    args.data_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["split.json", "split_summary.csv", "split_users.parquet"]:
        current_path = args.data_dir / filename
        backup_path = args.data_dir / f"previous_{filename}"
        if current_path.exists() and not backup_path.exists():
            shutil.copyfile(current_path, backup_path)

    shutil.copyfile(args.master_split, args.data_dir / "master_split_lesha.csv")
    save_user_split_tables(prepared, split, args.data_dir)
    summary_df.write_csv(args.data_dir / "master_split_import_summary.csv")


if __name__ == "__main__":
    main()
