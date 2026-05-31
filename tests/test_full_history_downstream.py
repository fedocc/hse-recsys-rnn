from math import isclose
import unittest

import polars as pl

from src.data import PreparedSequences
from src.downstream import build_full_history_baseline_features, build_meta_for_full_history


class FullHistoryDownstreamTest(unittest.TestCase):
    def test_build_meta_for_full_history_uses_duration_labels_without_prefix_filter(self):
        base_meta = pl.DataFrame(
            {
                "appmetrica_device_id": ["u1", "u2", "u3"],
                "all_timestamps": [[0, 86_400 * 8], [0, 86_400 * 3], [86_400, 86_400 * 20]],
                "all_sessions": [[1, 2], [1, 1], [3, 4]],
                "last_event_ts": [86_400 * 8, 86_400 * 3, 86_400 * 20],
            }
        )

        meta, summary = build_meta_for_full_history(
            base_meta,
            horizons={"retention_7d": 7, "retention_14d": 14},
        )

        self.assertEqual(
            meta.select("appmetrica_device_id", "history_len", "history_session_count").to_dicts(),
            [
                {"appmetrica_device_id": "u1", "history_len": 2, "history_session_count": 2},
                {"appmetrica_device_id": "u2", "history_len": 2, "history_session_count": 1},
                {"appmetrica_device_id": "u3", "history_len": 2, "history_session_count": 2},
            ],
        )
        self.assertEqual(
            meta.select("appmetrica_device_id", "retention_7d", "retention_14d").to_dicts(),
            [
                {"appmetrica_device_id": "u1", "retention_7d": 1, "retention_14d": 0},
                {"appmetrica_device_id": "u2", "retention_7d": 0, "retention_14d": 0},
                {"appmetrica_device_id": "u3", "retention_7d": 1, "retention_14d": 1},
            ],
        )
        self.assertEqual(
            summary.select("label", "valid_users").to_dicts(),
            [
                {"label": "retention_7d", "valid_users": 3},
                {"label": "retention_14d", "valid_users": 3},
            ],
        )


    def test_build_full_history_baseline_features_counts_last_window_shares(self):
        prepared = PreparedSequences(
            user_ids=["u1", "u2"],
            user_sequences=[[1, 2, 2, 3], [1, 1, 3]],
            event2idx={"a": 1, "b": 2, "c": 3},
        )
        train_prepared = PreparedSequences(
            user_ids=["u1"],
            user_sequences=[[1, 2, 2, 3]],
            event2idx=prepared.event2idx,
        )
        split_users = pl.DataFrame({"appmetrica_device_id": ["u1", "u2"], "split": ["train", "test"]})
        meta = pl.DataFrame(
            {
                "appmetrica_device_id": ["u1", "u2"],
                "history_session_count": [2, 1],
                "history_active_days": [2, 1],
                "history_span_days": [3.0, 1.0],
                "retention_7d": [0, 0],
            }
        )

        features, feature_cols, summary = build_full_history_baseline_features(
            prepared,
            train_prepared,
            split_users,
            meta,
            max_history_len=3,
            top_k_events=2,
            horizons={"retention_7d": 7},
        )

        row = features.filter(pl.col("appmetrica_device_id") == "u1").to_dicts()[0]
        self.assertEqual(row["history_len_feature"], 3)
        self.assertEqual(row["unique_event_count"], 2)
        self.assertTrue(isclose(row["repeat_ratio"], 1 / 3))
        self.assertTrue(isclose(row["share_b"], 2 / 3))
        self.assertTrue(isclose(row["share_c"], 1 / 3))
        self.assertNotIn("history_span_days", feature_cols)
        self.assertTrue(set(["history_session_count", "history_active_days"]).issubset(feature_cols))
        self.assertEqual(summary.to_dicts()[0]["history_window"], "last_3")


if __name__ == "__main__":
    unittest.main()
