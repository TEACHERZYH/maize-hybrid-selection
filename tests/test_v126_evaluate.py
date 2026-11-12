"""独立评价器：身份、预算积分、标签单位与进程隔离验收。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_v126.data import FRACTIONS, ROLE_SEEDS, S_REF, ids_sha256, sha256
from analysis_v126.evaluate import evaluate_role, evaluate_year, normalized_auc


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fixture(root, *, counts=(107, 107), modes=("good", "bad", "bad", "good")):
    data, predictions = root / "2019", root / "predictions"
    data.mkdir()
    ids, env, hybrid, audit_indices, query_indices, initial = [], [], [], [], [], []
    for e, n in enumerate(counts):
        start = len(ids)
        ids.extend(json.dumps([f"E{e}", f"G{i:03d}"], separators=(",", ":")) for i in range(n))
        hybrid.extend(f"G{i:03d}" for i in range(n))
        env.extend([e] * n)
        audit_count = (3 * n + 9) // 10
        audit_indices.extend(range(start, start + audit_count))
        query_indices.extend(range(start + audit_count, start + n))
        initial.extend(ids[start + audit_count:start + audit_count + 2])
    ids, env, hybrid = np.asarray(ids), np.asarray(env), np.asarray(hybrid)
    query_ids = [ids[i] for i in query_indices]
    query_ids = initial + [key for key in query_ids if key not in initial]
    audit_ids = ids[audit_indices]
    x = np.zeros((len(ids), 2), dtype=np.float32)
    target = {"x": x, "env": env, "combo_ids": ids, "hybrid_ids": hybrid,
              "genetic_pc": np.zeros((len(ids), 2), dtype=np.float64), "env_names": np.asarray([f"E{e}" for e in range(len(counts))]),
              "outer_year": np.asarray(2019), "x_center": np.zeros(2, dtype=np.float32), "x_scale": np.ones(2, dtype=np.float32)}
    np.savez(data / "target_features.npz", **target)
    center = 12.0
    audit_env = env[audit_indices]
    z = np.zeros(len(audit_indices), dtype=np.float64)
    for e in range(len(counts)):
        mask = audit_env == e
        z[mask] = np.arange(mask.sum()) / 10 + e * 10
    for seed in ROLE_SEEDS:
        budgets = {fraction: int(round(float(fraction) * 100)) * len(query_ids) // 100 for fraction in FRACTIONS}
        sources = [{"fraction": fraction, "budget": budgets[fraction], "source": "合成验证",
                    "sha256": "synthetic", "selected_ids_sha256": ids_sha256(query_ids[:budgets[fraction]])} for fraction in FRACTIONS]
        role = {"year": 2019, "seed": seed, "query_ids": query_ids, "audit_ids": audit_ids.tolist(),
                "initial_ids": initial, "budgets": {"0.0": 0, "1.0": len(query_ids), **budgets}, "M_sources": sources}
        role_path = data / "roles" / f"s{seed}" / "role.json"
        write_json(role_path, role)
        eval_dir = data / "evaluation" / f"s{seed}"
        eval_dir.mkdir(parents=True)
        np.savez(eval_dir / "audit_labels.npz", combo_ids=audit_ids, env=audit_env, hybrid_ids=hybrid[audit_indices],
                 y_raw=np.asarray(center + S_REF * z, dtype=np.float64), target_indices=np.asarray(audit_indices),
                 sref=np.asarray(S_REF), center=np.asarray(center))
        pred_dir = predictions / f"s{seed}"
        pred_dir.mkdir(parents=True)
        for fraction, mode in zip(FRACTIONS, modes):
            budget = budgets[fraction]
            support = {"year": 2019, "seed": seed, "fraction": fraction, "budget": budget,
                       "ids": query_ids[:budget], "labels": [0.0] * budget, "center": center, "sref": S_REF,
                       "label_unit": "historical_standardized", "role_sha256": sha256(role_path),
                       "ids_sha256": ids_sha256(query_ids[:budget]), "target_ids_sha256": ids_sha256(ids.tolist())}
            support_path = role_path.parent / f"support_b{fraction}.json"
            write_json(support_path, support)
            p = z.copy() if mode == "good" else -z if mode == "bad" else np.zeros(len(z))
            pred_path = pred_dir / f"b{fraction}.npz"
            np.savez(pred_path, combo_ids=audit_ids, prediction=p)
            write_json(pred_dir / f"b{fraction}.receipt.json", {
                "year": 2019, "seed": seed, "fraction": fraction, "budget": budget,
                "role_sha256": sha256(role_path), "support_sha256": sha256(support_path),
                "prediction_sha256": sha256(pred_path), "label_unit": "historical_standardized"})
    return data, predictions


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_actual_noninteger_budget_axis_and_single_standardization(self):
        data, predictions = fixture(self.root)
        result = evaluate_role(data, predictions / "s11", 11)
        expected = (1.3 * 15 + 2.6 * 30 + 1.3 * 29) / 74
        self.assertAlmostEqual(result["score"], expected, places=12)
        self.assertGreater(abs(result["score"] - 1.82), .001)
        self.assertAlmostEqual(result["budgets"][1]["macro_standardized_regret"], 2.6, places=12)
        self.assertEqual([row["budget"] for row in result["budgets"]], [14, 29, 59, 88])
        self.assertNotIn("y_raw", json.dumps(result))
        self.assertNotIn('"labels"', json.dumps(result))
        with self.assertRaises(ValueError):
            normalized_auc([1, 2, 3], [1, 2, 3], 100)

    def test_prediction_order_does_not_change_score_or_environment_metrics(self):
        data, predictions = fixture(self.root)
        before = evaluate_role(data, predictions / "s11", 11)
        rng = np.random.default_rng(126)
        for fraction in FRACTIONS:
            path = predictions / "s11" / f"b{fraction}.npz"
            with np.load(path) as archive:
                ids, prediction = archive["combo_ids"], archive["prediction"]
            perm = rng.permutation(len(ids))
            np.savez(path, combo_ids=ids[perm], prediction=prediction[perm])
            receipt_path = path.with_suffix(".receipt.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["prediction_sha256"] = sha256(path)
            write_json(receipt_path, receipt)
        after = evaluate_role(data, predictions / "s11", 11)
        self.assertEqual(before["score"], after["score"])
        self.assertEqual([x["environment_metrics"] for x in before["budgets"]], [x["environment_metrics"] for x in after["budgets"]])

    def test_environment_equal_weights_and_constant_prediction_missing_correlation(self):
        data, predictions = fixture(self.root, counts=(107, 151), modes=("tie",) * 4)
        result = evaluate_role(data, predictions / "s11", 11)
        # 33个审计品种取前7；46个审计品种取前10。按ID打破预测并列选出低产端。
        self.assertAlmostEqual(result["score"], (2.6 + 3.6) / 2)
        self.assertEqual(result["budgets"][0]["undefined_spearman_environments"], 2)
        self.assertIsNone(result["budgets"][0]["macro_spearman"])

    def test_reject_mismatched_receipt_budget_and_prediction_units(self):
        data, predictions = fixture(self.root)
        receipt_path = predictions / "s11" / "b0.1.receipt.json"
        original = json.loads(receipt_path.read_text(encoding="utf-8"))
        for key, value in (("budget", 15), ("support_sha256", "wrong"), ("label_unit", "t/ha"), ("seed", 23)):
            write_json(receipt_path, {**original, key: value})
            with self.assertRaises(ValueError):
                evaluate_role(data, predictions / "s11", 11)
        write_json(receipt_path, original)
        path = predictions / "s11" / "b0.1.npz"
        with np.load(path) as archive:
            ids, prediction = archive["combo_ids"].copy(), archive["prediction"].copy()
        ids[0] = ids[1]
        np.savez(path, combo_ids=ids, prediction=prediction)
        write_json(receipt_path, {**original, "prediction_sha256": sha256(path)})
        with self.assertRaises(ValueError):
            evaluate_role(data, predictions / "s11", 11)

    def test_five_role_average_and_independent_cli(self):
        data, predictions = fixture(self.root)
        # 一个角色全为正确预测，其余四角色保持原曲线；不得混为一个样本池。
        for fraction in FRACTIONS:
            path = predictions / "s11" / f"b{fraction}.npz"
            with np.load(predictions / "s11" / "b0.1.npz") as archive:
                ids, prediction = archive["combo_ids"].copy(), archive["prediction"].copy()
            np.savez(path, combo_ids=ids, prediction=prediction)
            receipt_path = path.with_suffix(".receipt.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            write_json(receipt_path, {**receipt, "prediction_sha256": sha256(path)})
        result = evaluate_year(data, predictions)
        expected = (1.3 * 15 + 2.6 * 30 + 1.3 * 29) / 74 * 4 / 5
        self.assertAlmostEqual(result["score"], expected, places=12)
        output = self.root / "score.json"
        completed = subprocess.run([sys.executable, str(ROOT / "analysis_v126/evaluate.py"), "--data-dir", str(data),
                                    "--predictions", str(predictions), "--output", str(output)],
                                   capture_output=True, text=True, encoding="utf-8", check=True)
        summary = json.loads(completed.stdout)
        self.assertEqual(set(summary), {"state", "year", "score", "macro_retention"})
        self.assertAlmostEqual(summary["score"], expected, places=12)
        self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
