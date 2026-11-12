"""外层预测的合成接口测试；不运行真实年份训练或评价。"""
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis_v126.data import FRACTIONS, ROLE_SEEDS, S_REF, ids_sha256, load_support, read_json, sha256
from analysis_v126.model import ResponseTabM, adapt_predict
from analysis_v126.outer_predict import (
    METHODS, ROOT, SourceContract, load_bound_model,
    planned_identities, predict_methods, validate_role, validate_stopping_contract,
    validate_support_request,
)
from analysis_v126.train import scientific_hashes, write_json


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_data(directory, year=2020):
    directory.mkdir(parents=True)
    rng = np.random.default_rng(9)
    ids = [f"c{i}" for i in range(12)]
    target = {"x": rng.normal(size=(12, 3)).astype(np.float32), "env": np.repeat([0, 1], 6),
              "combo_ids": np.asarray(ids), "hybrid_ids": np.asarray([f"h{i % 6}" for i in range(12)]),
              "genetic_pc": rng.normal(size=(12, 2)).astype(np.float64),
              "env_names": np.asarray(["e0", "e1"]), "outer_year": np.asarray(year),
              "x_center": np.zeros(3), "x_scale": np.ones(3)}
    history = {**target, "year": np.full(12, year - 1), "y": rng.normal(size=12).astype(np.float32),
               "center": np.asarray(8.0), "sref": np.asarray(S_REF)}
    np.savez(directory / "history.npz", **history)
    np.savez(directory / "target_features.npz", **target)
    receipt = {"state": "PASS", "year": year, "history_file_sha256": sha256(directory / "history.npz"),
               "target_file_sha256": sha256(directory / "target_features.npz"), "prepared_sha256": "new-prepared",
               "environment_availability_sha256": "new-availability"}
    write_json(directory / "export_receipt.json", receipt)
    budget_counts = {"0.0": 0, "0.1": 1, "0.2": 2, "0.4": 4, "0.6": 6, "1.0": 10}
    for seed in ROLE_SEEDS:
        role_path = directory / "roles" / f"s{seed}" / "role.json"
        role = {"year": year, "seed": seed, "query_ids": ids[:10], "audit_ids": ids[10:],
                "initial_ids": ids[:1], "budgets": budget_counts, "M_sources": []}
        for fraction in FRACTIONS:
            selected = ids[:budget_counts[fraction]]
            role["M_sources"].append({"fraction": fraction, "budget": len(selected),
                                      "selected_ids_sha256": ids_sha256(selected), "source": "unused.json", "sha256": "M"})
        write_json(role_path, role)
        for fraction, count in budget_counts.items():
            selected = ids[:count]
            write_json(role_path.parent / f"support_b{fraction}.json", {
                "year": year, "seed": seed, "fraction": fraction, "budget": count, "ids": selected,
                "labels": np.linspace(1, 3, count).tolist(), "center": 8.0, "sref": S_REF,
                "label_unit": "historical_standardized", "ids_sha256": ids_sha256(selected),
                "target_ids_sha256": ids_sha256(ids), "role_sha256": sha256(role_path),
            })
    # 故意放入无法解析的隐藏文件；预测进程不能读取它。
    (directory / "evaluation").mkdir()
    (directory / "evaluation" / "audit_labels.json").write_text("HIDDEN_MUST_NOT_BE_READ", encoding="utf-8")
    return history, target


def make_stopping(branch="ordinary", epoch=2):
    dev = {"state": "COMPLETED", "outer_year": 2019, "seed": 1009, "stage": branch,
           "best_epoch": epoch, "completed_epoch": max(5, epoch + 5),
           "stopping": {"stop": True, "no_improvement": 5}}
    outer = {"state": "COMPLETED", "outer_year": 2020, "seed": 1009, "stage": branch,
             "best_epoch": epoch, "completed_epoch": epoch}
    return dev, outer


def test_method_support_semantics_and_matrix():
    assert METHODS["C1"] == ("ordinary", "intercept", FRACTIONS)
    assert METHODS["N1_INT"] == ("N1", "intercept", FRACTIONS)
    assert METHODS["N1_ZERO"][2] == ("0.0",)
    assert METHODS["C2_ALL"][2] == ("1.0",)
    identities = planned_identities(ROOT / "design_v126/planned_runs.csv", ("C1", "C2_ZERO", "C2_ALL"), 2020, 1009, "ordinary")
    assert len(identities) == 30
    assert identities[("C2_ZERO", 11, "0.0")] == "C2_ZERO_y2020_r11_t1009_b0"
    with pytest.raises(ValueError):
        planned_identities(ROOT / "design_v126/planned_runs.csv", ("N1",), 2020, 1009, "ordinary")


@pytest.mark.parametrize("change", ["unfinished", "wrong_seed", "wrong_epoch", "wrong_branch", "fake_stop", "N1_zero"])
def test_reject_invalid_stopping_binding(change):
    dev, outer = make_stopping()
    branch = "ordinary"
    if change == "unfinished":
        dev["state"] = "CHECKPOINT_READY"
    elif change == "wrong_seed":
        outer["seed"] = 2027
    elif change == "wrong_epoch":
        outer["completed_epoch"] = 1
    elif change == "wrong_branch":
        dev["stage"] = "N1"
    elif change == "fake_stop":
        dev["stopping"]["no_improvement"] = 4
    else:
        branch = "N1"
        dev, outer = make_stopping(branch, 0)
    with pytest.raises(ValueError):
        validate_stopping_contract(dev, outer, 2020, 1009, branch)


def test_ordinary_zero_epoch_allowed():
    dev, outer = make_stopping(epoch=0)
    assert validate_stopping_contract(dev, outer, 2020, 1009, "ordinary") == 0


def test_role_and_support_exactness(tmp_path):
    _, target = make_data(tmp_path / "data")
    role_path = tmp_path / "data/roles/s11/role.json"
    role = read_json(role_path)
    support = load_support(role_path.parent / "support_b1.0.json", target, role_path)
    validate_role(role, target, 2020, 11)
    validate_support_request(support, role, "1.0")
    with pytest.raises(ValueError):
        validate_support_request(support, role, "0.0")
    support["ids"] = support["ids"][:-1]
    with pytest.raises(ValueError):
        validate_support_request(support, role, "1.0")
    role["query_ids"].append(role["audit_ids"][0])
    with pytest.raises(ValueError):
        validate_role(role, target, 2020, 11)


def test_checkpoint_binds_data_projection_seed_and_epoch(tmp_path):
    history, target = make_data(tmp_path / "data")
    source = SourceContract(tmp_path / "data", 2020)
    model = ResponseTabM(3, 1009)
    projection = tmp_path / "projection.npy"
    np.save(projection, model.projection.detach().numpy())
    dev, outer = make_stopping()
    dev_path, outer_path = tmp_path / "dev.json", tmp_path / "stage.json"
    write_json(dev_path, dev)
    write_json(outer_path, outer)
    hashes = scientific_hashes(source.data_dir, projection, "ordinary")
    hashes["development_stage_result.json"] = sha256(dev_path)
    payload = {"format": "v1.26", "stage": "ordinary", "outer_year": 2020, "seed": 1009,
               "epoch": 2, "steps_per_epoch": 100, "input_dim": 3, "model": model.state_dict(),
               "projection_sha256": model.projection_sha256, "scientific_files_sha256": hashes}
    path = tmp_path / "checkpoint_002.pt"
    torch.save(payload, path)
    write_json(path.with_suffix(".state.json"), {"checkpoint_sha256": sha256(path)})
    loaded, binding = load_bound_model(path, dev_path, outer_path, source, 2020, 1009, "ordinary", projection)
    assert binding["E_m"] == 2 and loaded.projection_sha256 == model.projection_sha256
    payload["scientific_files_sha256"]["analysis_v126/train.py"] = "different_loss"
    torch.save(payload, path)
    write_json(path.with_suffix(".state.json"), {"checkpoint_sha256": sha256(path)})
    with pytest.raises(ValueError):
        load_bound_model(path, dev_path, outer_path, source, 2020, 1009, "ordinary", projection)
    payload["scientific_files_sha256"]["analysis_v126/train.py"] = sha256(ROOT / "analysis_v126/train.py")
    payload["epoch"] = 3
    torch.save(payload, path)
    write_json(path.with_suffix(".state.json"), {"checkpoint_sha256": sha256(path)})
    with pytest.raises(ValueError):
        load_bound_model(path, dev_path, outer_path, source, 2020, 1009, "ordinary", projection)


def test_2024_new_source_only_and_hidden_file_not_opened(tmp_path, monkeypatch):
    data = tmp_path / "new2024"
    make_data(data, 2024)
    matrix = ROOT / "design_v126/revisions/planned_runs_v1.26-R2.csv"
    public_files = {p.relative_to(data).as_posix(): sha256(p) for p in data.rglob("*") if p.is_file() and "evaluation" not in p.parts}
    public_files["evaluation/audit_labels.json"] = "not-read-by-predictor"
    registry = {"state": "PASS_EXECUTION_SOURCES", "version": "v1.26-R2", "year": 2024,
                "required_2024_data_dir": str(data), "new_export_directory": str(data), "new_export_files": public_files,
                "revised_matrix": {"path": str(matrix), "sha256": sha256(matrix)},
                "prepared": {"sha256": "new-prepared"}, "availability": {"sha256": "new-availability"}}
    registry_path = tmp_path / "registry.json"
    write_json(registry_path, registry)
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert "evaluation" not in path.parts, "预测进程访问隐藏标签"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    contract = SourceContract(data, 2024, registry_path)
    assert contract.registry_sha256 == sha256(registry_path)
    with pytest.raises(ValueError):
        SourceContract(tmp_path / "old2024", 2024, registry_path)
    with pytest.raises(ValueError):
        contract.verify("evaluation/audit_labels.json")
    registry["new_export_files"]["history.npz"] = "wrong"
    write_json(registry_path, registry)
    with pytest.raises(ValueError):
        SourceContract(data, 2024, registry_path)


def test_prediction_matches_direct_joint_solution_and_reuses_binding(tmp_path, monkeypatch):
    history, target = make_data(tmp_path / "data")
    source = SourceContract(tmp_path / "data", 2020)
    methods = ("C1", "C2", "C2_ZERO", "C2_ALL")
    model = ResponseTabM(3, 1009).eval()
    binding = {"year": 2020, "training_seed": 1009, "checkpoint_family": "ordinary", "E_m": 2}
    identities = planned_identities(source.matrix_path, methods, 2020, 1009, "ordinary")
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert "evaluation" not in path.parts, "预测进程访问隐藏标签"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    result = predict_methods(model, history, target, source, tmp_path / "out", methods, binding, identities, chunk_size=5)
    assert result["count"] == 50 and result["state"] == "PREDICTIONS_READY"
    role_path = source.data_dir / "roles/s11/role.json"
    with torch.no_grad():
        base, phi = model(torch.as_tensor(target["x"]))
        for method, fraction in (("C1", "0.4"), ("C2", "0.4"), ("C2_ZERO", "0.0"), ("C2_ALL", "1.0")):
            support = load_support(role_path.parent / f"support_b{fraction}.json", target, role_path)
            expected, _ = adapt_predict(base, phi, torch.as_tensor(target["env"]), torch.as_tensor(support["indices"]),
                                        torch.as_tensor(support["y"]), mode=METHODS[method][1])
            path = tmp_path / "out" / method / "s11" / f"b{fraction}.npz"
            with np.load(path) as saved:
                assert set(saved.files) == {"combo_ids", "prediction"}
                np.testing.assert_array_equal(saved["combo_ids"], target["combo_ids"][-2:])
                np.testing.assert_allclose(saved["prediction"], expected.mean(1).numpy()[-2:], atol=1e-6, rtol=1e-6)
            receipt = read_json(path.with_suffix(".receipt.json"))
            assert receipt["prediction_sha256"] == sha256(path)
    path = tmp_path / "out/C2_ZERO/s11/b0.0.npz"
    original_hash = sha256(path)
    predict_methods(model, history, target, source, tmp_path / "out", ("C2_ZERO",), binding, identities, chunk_size=5)
    assert sha256(path) == original_hash
    with pytest.raises(ValueError):
        predict_methods(model, history, target, source, tmp_path / "out", methods,
                        {**binding, "training_seed": 2027}, identities)
