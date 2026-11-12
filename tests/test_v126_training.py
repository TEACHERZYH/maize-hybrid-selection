"""v1.26 训练合成测试；短步只检查代码，不构成候选性能实验。"""

import copy

import numpy as np
import pytest
import torch

from analysis_v126.model import ResponseTabM
from analysis_v126.train import (
    completion_state, env_groups, equal_environment_loss, load_checkpoint, make_optimizer,
    optimization_step, sample_episode, sample_pretrain, save_checkpoint,
    update_stopping, validate_prediction_metadata,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def history_fixture():
    rng = np.random.default_rng(126)
    env = np.repeat(np.arange(4), 105)
    n = len(env)
    return {"x": rng.normal(size=(n, 6)).astype(np.float32), "y": rng.normal(size=n).astype(np.float32),
            "env": env, "year": np.repeat([2017, 2018], 210),
            "combo_ids": np.asarray([f"c{i:04d}" for i in range(n)]),
            "hybrid_ids": np.asarray([f"h{i % 105:03d}" for i in range(n)]),
            "genetic_pc": rng.normal(size=(n, 4))}


def test_environment_equal_weight_and_centered_loss():
    prediction = torch.tensor([[3., 3.], [1., 1.], [3., 3.], [5., 5.]])
    label = torch.zeros(4)
    env = torch.tensor([0, 1, 1, 1])
    expected = (9 + (1 + 9 + 25) / 3) / 2
    assert equal_environment_loss(prediction, label, env).item() == pytest.approx(expected)
    expected_centered = expected + ((1 - 3) ** 2 + (3 - 3) ** 2 + (5 - 3) ** 2) / 3 / 2
    assert equal_environment_loss(prediction, label, env, centered=True).item() == pytest.approx(expected_centered)


def test_pretraining_env_and_per_env_sample_limits():
    history = history_fixture()
    a = sample_pretrain(history, 2019, 1009, 4)
    b = sample_pretrain(history, 2019, 1009, 4)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 4 * 64 and len(np.unique(a)) == len(a)
    assert all(len(indices) == 64 for indices in env_groups(history, a).values())


def test_episode_budget_bridge_and_target_isolation():
    history = history_fixture()
    a = sample_episode(history, 2019, 1009, 11)
    b = sample_episode(history, 2019, 1009, 11)
    for field in ("pool", "query", "target", "support", "initial", "loss_target"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert a.year in (2017, 2018)
    assert len(a.pool) == 210 and len(a.target) == 2 * 32
    assert len(a.support) == int(a.fraction * len(a.query))
    assert set(a.initial).issubset(a.support) and len(a.initial) == 4
    assert set(a.support).issubset(a.query) and not set(a.support) & set(a.target)
    assert set(a.loss_target).issubset(a.target)
    assert len(set(a.query) | set(a.target)) == len(a.pool)
    assert np.all(history["year"][a.pool] < 2019)


@pytest.mark.parametrize("stage", ["pretrain", "ordinary", "centered", "N1"])
def test_one_synthetic_training_step_finite(stage):
    history = history_fixture()
    model = ResponseTabM(6, 1009)
    optimizer = make_optimizer(model, stage)
    before = [p.detach().clone() for p in model.parameters()]
    metrics = optimization_step(model, optimizer, history, stage, 2019, 1009, 0, chunk_size=128, diagnose=True)
    assert np.isfinite(metrics["loss"]) and np.isfinite(metrics["gradient_norm_before_clip"])
    assert any(not torch.equal(p.detach(), b) for p, b in zip(model.parameters(), before))
    assert metrics["seconds"] > 0


def test_checkpoint_resume_preserves_optimizer_and_next_update(tmp_path):
    history = history_fixture()
    model = ResponseTabM(6, 1009)
    optimizer = make_optimizer(model, "pretrain")
    optimization_step(model, optimizer, history, "pretrain", 2019, 1009, 0, chunk_size=128)
    checkpoint = tmp_path / "checkpoint.pt"
    hashes = {"synthetic_history": "test-only"}
    save_checkpoint(checkpoint, model, optimizer, stage="pretrain", epoch=1, seed=1009, outer_year=2019, hashes=hashes, stopping={})
    replica = ResponseTabM(6, 1009)
    replica_optimizer = make_optimizer(replica, "pretrain")
    payload = load_checkpoint(checkpoint, replica, replica_optimizer, stage="pretrain", seed=1009, outer_year=2019, hashes=hashes)
    assert payload["epoch"] == 1 and not payload["epoch_finalized"]
    a = optimization_step(model, optimizer, history, "pretrain", 2019, 1009, 1, chunk_size=128)
    b = optimization_step(replica, replica_optimizer, history, "pretrain", 2019, 1009, 1, chunk_size=128)
    assert a["loss"] == b["loss"]
    for p, q in zip(model.parameters(), replica.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    with pytest.raises(ValueError, match="哈希"):
        load_checkpoint(checkpoint, replica, replica_optimizer, stage="pretrain", seed=1009, outer_year=2019, hashes={"changed": "no"})


def test_three_B_branches_share_A_model_optimizer_and_rng(tmp_path):
    history = history_fixture()
    model = ResponseTabM(6, 1009)
    optimizer = make_optimizer(model, "pretrain")
    optimization_step(model, optimizer, history, "pretrain", 2019, 1009, 0, chunk_size=128)
    # 仅模拟验收字段，临时夹具不是完成20epoch的科学记录。
    path = tmp_path / "synthetic_A.pt"
    save_checkpoint(path, model, optimizer, stage="pretrain", epoch=20, seed=1009, outer_year=2019, hashes={}, stopping={})
    loaded = []
    for stage in ("ordinary", "centered", "N1"):
        branch = ResponseTabM(6, 1009)
        opt = make_optimizer(branch, stage)
        load_checkpoint(path, branch, opt, stage=stage, seed=1009, outer_year=2019, hashes={}, inherit_A=True)
        assert opt.param_groups[0]["lr"] == 3e-4
        for p, q in zip(branch.parameters(), model.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        loaded.append(copy.deepcopy(opt.state_dict()))
    for other in loaded[1:]:
        for key, state in loaded[0]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, other["state"][key][name], rtol=0, atol=0)


def test_stopping_keeps_earliest_near_tie_and_waits_five_epochs():
    state = update_stopping({}, 0.5, 0)
    for epoch in range(1, 5):
        state = update_stopping(state, 0.5 - 1e-9, epoch)
        assert not state["stop"] and state["best_epoch"] == 0
    state = update_stopping(state, 0.5, 5)
    assert state["stop"] and state["best_epoch"] == 0
    state = update_stopping(state, 0.49, 6)
    assert state["best_epoch"] == 6 and state["no_improvement"] == 0 and not state["stop"]


def test_partial_development_is_not_final_stop_contract():
    assert completion_state("pretrain", 2019, 5, {}) == "CHECKPOINT_READY"
    assert completion_state("pretrain", 2019, 20, {}) == "COMPLETED"
    assert completion_state("N1", 2019, 5, {"stop": False}) == "CHECKPOINT_READY"
    assert completion_state("N1", 2019, 5, {"stop": True}) == "COMPLETED"
    assert completion_state("N1", 2019, 30, {"stop": False}) == "COMPLETED"


def test_prediction_metadata_rejects_mixed_year_encoding_or_label_scale():
    history = {"outer_year": np.asarray(2019), "x": np.zeros((3, 2)), "env_names": np.asarray(["a", "b"]),
               "x_center": np.zeros(2), "x_scale": np.ones(2), "center": np.asarray(2.), "sref": np.asarray(3.)}
    target = copy.deepcopy(history)
    support = {"receipt": {"center": 2., "sref": 3.}}
    validate_prediction_metadata(history, target, support)
    for key, wrong in [("outer_year", np.asarray(2020)), ("env_names", np.asarray(["b", "a"])),
                       ("x_center", np.ones(2)), ("x_scale", np.ones(2) * 2)]:
        bad = copy.deepcopy(target)
        bad[key] = wrong
        with pytest.raises(ValueError):
            validate_prediction_metadata(history, bad, support)
    with pytest.raises(ValueError, match="支持标签"):
        validate_prediction_metadata(history, target, {"receipt": {"center": 5., "sref": 3.}})
