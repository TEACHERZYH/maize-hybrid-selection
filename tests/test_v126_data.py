"""v1.26 数据角色和标签权限的最小正确性检查。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_v126.data import (S_REF, eligible_episode_years, ids_sha256, load_history,
                               load_support, load_target_features, load_training_inputs, sha256)
from operations.prepare_v126_data import parse_source_date


class DataIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        common = dict(x=np.zeros((4, 2), dtype=np.float32), env=np.asarray([0, 0, 1, 1]),
                      combo_ids=np.asarray(["a", "b", "c", "d"]), hybrid_ids=np.asarray(["A", "B", "A", "B"]),
                      genetic_pc=np.zeros((4, 2), dtype=np.float64), env_names=np.asarray(["E1", "E2"]),
                      outer_year=np.asarray(2020), x_center=np.zeros(2), x_scale=np.ones(2))
        self.target_arrays = common
        np.savez(self.path / "target.npz", **common)
        historical = {**common, "combo_ids": np.asarray(["ha", "hb", "hc", "hd"]),
                      "y": np.zeros(4, dtype=np.float32), "year": np.full(4, 2019),
                      "center": np.asarray(8.0), "sref": np.asarray(S_REF)}
        self.history_arrays = historical
        np.savez(self.path / "history.npz", **historical)
        role = {"seed": 11, "year": 2020, "query_ids": ["a", "c"], "audit_ids": ["b", "d"],
                "initial_ids": ["a", "c"], "budgets": {"1.0": 2, "0.0": 0}}
        self.role_path = self.path / "role.json"
        self.role_path.write_text(json.dumps(role), encoding="utf-8")
        self.support = {"seed": 11, "year": 2020, "fraction": "1.0", "budget": 2,
                        "ids": ["a", "c"], "labels": [1.0, 2.0], "center": 8.0, "sref": S_REF,
                        "role_sha256": sha256(self.role_path), "ids_sha256": ids_sha256(["a", "c"]),
                        "target_ids_sha256": ids_sha256(common["combo_ids"].tolist()), "label_unit": "historical_standardized"}
        self.support_path = self.path / "support.json"
        self.save_support()

    def tearDown(self):
        self.tmp.cleanup()

    def save_support(self):
        self.support_path.write_text(json.dumps(self.support), encoding="utf-8")

    def test_only_allowed_labels_loaded(self):
        data = load_training_inputs(self.path / "history.npz", self.path / "target.npz", self.support_path, self.role_path, 2020)
        self.assertEqual(set(data), {"history", "target", "support"})
        self.assertNotIn("y", data["target"])
        np.testing.assert_equal(data["support"]["indices"], [0, 2])

    def test_audit_label_perturbation_cannot_enter_loader(self):
        target = load_target_features(self.path / "target.npz")
        before = load_support(self.support_path, target, self.role_path)["y"]
        np.savez(self.path / "audit_labels.npz", y_raw=np.asarray([9999.0, -9999.0]))
        after = load_support(self.support_path, target, self.role_path)["y"]
        np.testing.assert_array_equal(before, after)

    def test_forbids_target_label_field(self):
        np.savez(self.path / "target.npz", **self.target_arrays, current_values=np.zeros(4))
        with self.assertRaisesRegex(ValueError, "标签隔离"):
            load_target_features(self.path / "target.npz")

    def test_forbids_future_history(self):
        self.history_arrays["year"][0] = 2020
        np.savez(self.path / "history.npz", **self.history_arrays)
        with self.assertRaisesRegex(ValueError, "未来"):
            load_history(self.path / "history.npz", 2020)

    def test_forbids_mismatched_outer_target_year(self):
        self.target_arrays["outer_year"] = np.asarray(2021)
        np.savez(self.path / "target.npz", **self.target_arrays)
        with self.assertRaisesRegex(ValueError, "目标年份"):
            load_training_inputs(self.path / "history.npz", self.path / "target.npz", self.support_path, self.role_path, 2020)

    def test_forbids_audit_support_even_if_hash_recomputed(self):
        self.support["ids"] = ["a", "b"]
        self.support["ids_sha256"] = ids_sha256(self.support["ids"])
        self.save_support()
        with self.assertRaisesRegex(ValueError, "审计"):
            load_support(self.support_path, load_target_features(self.path / "target.npz"), self.role_path)

    def test_zero_support_no_free_s0(self):
        self.support.update(ids=[], labels=[], budget=0, fraction="0.0", ids_sha256=ids_sha256([]))
        self.save_support()
        self.assertEqual(len(load_support(self.support_path, load_target_features(self.path / "target.npz"), self.role_path)["y"]), 0)

    def test_support_duplicate_rejected(self):
        self.support["ids"] = ["a", "a"]
        self.save_support()
        with self.assertRaisesRegex(ValueError, "重复"):
            load_support(self.support_path, load_target_features(self.path / "target.npz"), self.role_path)

    def test_episode_eligibility_does_not_filter_A(self):
        history = {"year": np.asarray([2018] * 199 + [2019] * 200),
                   "env": np.asarray([0] * 100 + [1] * 99 + [2] * 100 + [3] * 100)}
        self.assertEqual(eligible_episode_years(history), {2019: [2, 3]})
        self.assertEqual(len(history["year"]), 399)

    def test_soil_date_counterexample(self):
        self.assertLess("11/14/2024", "4/15/2024")
        self.assertGreater(parse_source_date("11/14/2024"), parse_source_date("4/15/2024"))
        self.assertEqual(parse_source_date("11/14/2024"), parse_source_date("2024-11-14"))


if __name__ == "__main__":
    unittest.main()
