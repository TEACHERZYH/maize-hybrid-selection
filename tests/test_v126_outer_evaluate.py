"""外层评价使用人为排序数据，禁止读取真实外层标签。"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from analysis_v126.data import FRACTIONS, ROLE_SEEDS, S_REF, ids_sha256, read_json, sha256
from analysis_v126.evaluate import evaluate_role
from analysis_v126.outer_evaluate import (
    BINDING_KEYS, METHODS, ROOT, evaluate_manifest, evaluate_reference_role, validate_manifest,
)
from test_v126_evaluate import fixture, write_json


def outer_fixture(root, *, year=2020, modes=("tie",) * 4):
    data, original = fixture(root, counts=(107, 151), modes=modes)
    with np.load(data / "target_features.npz") as saved:
        target = {key: saved[key] for key in saved.files}
    target["outer_year"] = np.asarray(year)
    np.savez(data / "target_features.npz", **target)
    np.savez(data / "history.npz", **target, year=np.full(len(target["x"]), year - 1),
             y=np.zeros(len(target["x"]), dtype=np.float32), center=np.asarray(12.), sref=np.asarray(S_REF))
    for seed in ROLE_SEEDS:
        role_path = data / "roles" / f"s{seed}" / "role.json"
        role = read_json(role_path)
        role["year"] = year
        write_json(role_path, role)
        for fraction in ("0.0", *FRACTIONS, "1.0"):
            support_path = role_path.parent / f"support_b{fraction}.json"
            if fraction in FRACTIONS:
                support = read_json(support_path)
            else:
                ids = [] if fraction == "0.0" else role["query_ids"]
                support = {"seed": seed, "fraction": fraction, "budget": len(ids), "ids": ids,
                           "labels": [0.] * len(ids), "center": 12., "sref": S_REF,
                           "label_unit": "historical_standardized", "ids_sha256": ids_sha256(ids),
                           "target_ids_sha256": ids_sha256(target["combo_ids"].tolist())}
            support.update(year=year, role_sha256=sha256(role_path))
            write_json(support_path, support)
            if fraction in FRACTIONS:
                receipt_path = original / f"s{seed}" / f"b{fraction}.receipt.json"
                receipt = read_json(receipt_path)
                receipt.update(year=year, role_sha256=sha256(role_path), support_sha256=sha256(support_path))
                write_json(receipt_path, receipt)
    output = root / "outer"
    shutil.copytree(original, output / "C2")
    for method, fraction in (("C2_ZERO", "0.0"), ("C2_ALL", "1.0")):
        for seed in ROLE_SEEDS:
            directory = output / method / f"s{seed}"
            directory.mkdir(parents=True)
            old_path = original / f"s{seed}/b0.1.npz"
            path = directory / f"b{fraction}.npz"
            shutil.copyfile(old_path, path)
            receipt = read_json(old_path.with_suffix(".receipt.json"))
            role_path = data / "roles" / f"s{seed}/role.json"
            support_path = role_path.parent / f"support_b{fraction}.json"
            receipt.update(fraction=fraction, budget=read_json(support_path)["budget"],
                           support_sha256=sha256(support_path), prediction_sha256=sha256(path))
            write_json(path.with_suffix(".receipt.json"), receipt)
    matrix = ROOT / ("design_v126/revisions/planned_runs_v1.26-R2.csv" if year == 2024 else "design_v126/planned_runs.csv")
    registry_path = None
    if year == 2024:
        registry_path = root / "sources_R2.json"
        write_json(registry_path, {"state": "PASS_EXECUTION_SOURCES", "version": "v1.26-R2", "year": year,
                                  "required_2024_data_dir": str(data.resolve()), "new_export_directory": str(data.resolve()),
                                  "revised_matrix": {"path": str(matrix), "sha256": sha256(matrix)},
                                  "new_export_files": {p.relative_to(data).as_posix(): sha256(p) for p in data.rglob("*") if p.is_file()}})
    binding = {key: "synthetic_" + key for key in BINDING_KEYS}
    binding.update(year=year, training_seed=1009, checkpoint_family="ordinary", E_m=2,
                   history_sha256=sha256(data / "history.npz"), target_features_sha256=sha256(data / "target_features.npz"),
                   member_reduction="mean_16", matrix_sha256=sha256(matrix),
                   execution_sources_sha256=sha256(registry_path) if registry_path else None)
    write_json(output / "execution_binding.json", binding)
    records = []
    for method in ("C2", "C2_ZERO", "C2_ALL"):
        for seed in ROLE_SEEDS:
            for fraction in METHODS[method][2]:
                path = output / method / f"s{seed}/b{fraction}.npz"
                budget_name = {"0.0": "0", "1.0": "1"}.get(fraction, fraction)
                run_id = f"{method}_y{year}_r{seed}_t1009_b{budget_name}"
                diagnostics_path = path.with_suffix(".diagnostics.json")
                write_json(diagnostics_path, {"synthetic": True})
                receipt = read_json(path.with_suffix(".receipt.json"))
                receipt.update(binding)
                receipt.update(run_id=run_id, method=method, mode=METHODS[method][1], diagnostics_sha256=sha256(diagnostics_path))
                write_json(path.with_suffix(".receipt.json"), receipt)
                records.append({"run_id": run_id, "prediction_path": str(path.resolve()),
                                "receipt_sha256": sha256(path.with_suffix(".receipt.json")), "seconds": 0.})
    manifest = {"state": "PREDICTIONS_READY", **binding, "methods": ["C2", "C2_ZERO", "C2_ALL"],
                "count": len(records), "records": records,
                "model_readable_files_sha256": {p.relative_to(data).as_posix(): sha256(p) for p in data.rglob("*")
                                               if p.is_file() and "evaluation" not in p.parts}}
    manifest_path = output / "predictions_C2_C2_ZERO_C2_ALL.json"
    write_json(manifest_path, manifest)
    return data, output, manifest_path, registry_path


def test_reference_same_units_environment_weights_and_cvar(tmp_path):
    data, output, _, _ = outer_fixture(tmp_path)
    main = evaluate_role(data, output / "C2/s11", 11)["budgets"][0]
    for method, fraction in (("C2_ZERO", "0.0"), ("C2_ALL", "1.0")):
        result = evaluate_reference_role(data, output / method / "s11", 11, fraction)
        for key in ("macro_standardized_regret", "macro_rmse", "macro_centered_rmse", "macro_retention",
                    "global_rmse", "macro_spearman", "undefined_spearman_environments", "environment_cvar_0.8", "environment_metrics"):
            assert result[key] == main[key]
        assert result["macro_standardized_regret"] == pytest.approx((2.6 + 3.6) / 2)
        assert result["environment_cvar_0.8"] == pytest.approx(3.6)
        assert result["actual_fraction"] == float(fraction)
        assert result["macro_spearman"] is None and result["undefined_spearman_environments"] == 2
        assert "score" not in result and "auc" not in result
        assert "y_raw" not in json.dumps(result) and '"labels"' not in json.dumps(result)


def test_reference_prediction_order_invariant(tmp_path):
    data, output, _, _ = outer_fixture(tmp_path, modes=("bad",) * 4)
    directory = output / "C2_ALL/s11"
    before = evaluate_reference_role(data, directory, 11, "1.0")
    path = directory / "b1.0.npz"
    with np.load(path) as saved:
        ids, prediction = saved["combo_ids"], saved["prediction"]
    permutation = np.random.default_rng(99).permutation(len(ids))
    np.savez(path, combo_ids=ids[permutation], prediction=prediction[permutation])
    receipt_path = path.with_suffix(".receipt.json")
    receipt = read_json(receipt_path)
    receipt["prediction_sha256"] = sha256(path)
    write_json(receipt_path, receipt)
    after = evaluate_reference_role(data, directory, 11, "1.0")
    assert before["environment_metrics"] == after["environment_metrics"]
    assert before["macro_standardized_regret"] == after["macro_standardized_regret"]


def test_manifest_four_budgets_reuse_original_and_refs_have_no_auc(tmp_path):
    data, output, manifest_path, _ = outer_fixture(tmp_path, modes=("good", "bad", "bad", "good"))
    result = evaluate_manifest(data, manifest_path)
    expected = [evaluate_role(data, output / "C2" / f"s{s}", s) for s in ROLE_SEEDS]
    assert result["methods"]["C2"]["roles"] == expected
    assert result["methods"]["C2"]["score"] == np.mean([r["score"] for r in expected])
    for method in ("C2_ZERO", "C2_ALL"):
        assert "score" not in result["methods"][method]
        assert result["methods"][method]["score_name"] == "single_budget_standardized_regret_no_auc"
        assert result["methods"][method]["macro_standardized_regret"] == pytest.approx(0.)
    assert result["state"] == "EXECUTION_VALID" and result["scientific_positive_claim_accepted"] is False


@pytest.mark.parametrize("field,value", [("training_seed", 2027), ("method", "C1"), ("checkpoint_sha256", "wrong"), ("mode", "intercept")])
def test_manifest_rejects_mixed_model_method_seed(tmp_path, field, value):
    data, output, manifest_path, _ = outer_fixture(tmp_path)
    path = output / "C2/s11/b0.1.receipt.json"
    receipt = read_json(path)
    receipt[field] = value
    write_json(path, receipt)
    manifest = read_json(manifest_path)
    manifest["records"][0]["receipt_sha256"] = sha256(path)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        validate_manifest(data, manifest_path)


def test_reference_rejects_bad_units_or_nonempty_zero(tmp_path):
    data, output, _, _ = outer_fixture(tmp_path)
    directory = output / "C2_ZERO/s11"
    receipt_path = directory / "b0.0.receipt.json"
    receipt = read_json(receipt_path)
    receipt["label_unit"] = "raw"
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        evaluate_reference_role(data, directory, 11, "0.0")
    support_path = data / "roles/s11/support_b0.0.json"
    support = read_json(support_path)
    support["ids"] = read_json(data / "roles/s11/role.json")["initial_ids"]
    support["budget"] = len(support["ids"])
    support["labels"] = [0.] * support["budget"]
    support["ids_sha256"] = ids_sha256(support["ids"])
    write_json(support_path, support)
    with pytest.raises(ValueError):
        evaluate_reference_role(data, directory, 11, "0.0")


def test_2024_registry_and_predictions_validated_before_hidden_files(tmp_path, monkeypatch):
    data, _, manifest_path, registry_path = outer_fixture(tmp_path, year=2024)
    assert validate_manifest(data, manifest_path, registry_path)["year"] == 2024
    old_directory = tmp_path / "old2024"
    shutil.copytree(data, old_directory)
    with pytest.raises(ValueError):
        validate_manifest(old_directory, manifest_path, registry_path)
    manifest = read_json(manifest_path)
    manifest["records"][0]["receipt_sha256"] = "wrong"
    write_json(manifest_path, manifest)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert "evaluation" not in path.parts, "预测身份未通过之前读取了隐藏标签"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError):
        validate_manifest(data, manifest_path, registry_path)
