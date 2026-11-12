"""v1.26 合成数据连通性、标签隔离和联合解检查；不检验性能优劣。"""

import copy

import numpy as np
import pytest
import torch

from analysis_v126.model import ResponseTabM, adapt_predict
from analysis_v126.vendor.tabm import TabM


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def synthetic():
    rng = torch.Generator().manual_seed(73)
    base = torch.randn((18, 3), generator=rng, dtype=torch.float64)
    phi = torch.tanh(torch.randn((18, 3, 8), generator=rng, dtype=torch.float64))
    env = torch.tensor([4] * 6 + [9] * 6 + [21] * 6)
    support = torch.tensor([0, 1, 3, 6, 7, 10, 12, 14, 16])
    labels = torch.randn(18, generator=rng, dtype=torch.float64)
    return base, phi, env, support, labels


@pytest.mark.parametrize("mode", ["response", "intercept"])
def test_joint_solution_matches_augmented_least_squares(mode):
    base, phi, env, support, labels = synthetic()
    prediction, diag = adapt_predict(base, phi, env, support, labels[support], mode=mode)
    values = np.unique(env.numpy())
    for member in range(base.shape[1]):
        onehot = np.column_stack([env.numpy() == e for e in values]).astype(float)
        centered = phi[:, member].numpy().copy()
        for e in values:
            centered[env.numpy() == e] -= centered[env.numpy() == e].mean(axis=0)
        design = np.column_stack([onehot, centered]) if mode == "response" else onehot
        regularizer = np.diag(np.sqrt([1.] * len(values) + ([10.] * 8 if mode == "response" else [])))
        augmented = np.row_stack([design[support.numpy()], regularizer])
        residual = labels[support].numpy() - base[support, member].numpy()
        beta = np.linalg.lstsq(augmented, np.r_[residual, np.zeros(design.shape[1])], rcond=None)[0]
        np.testing.assert_allclose(diag["beta"][member].numpy(), beta, atol=1e-12)
        np.testing.assert_allclose(prediction[:, member].numpy(), base[:, member].numpy() + design @ beta, atol=1e-12)
    assert torch.isfinite(diag["ridge_condition"]).all()
    assert (diag["effective_df"] >= 0).all()


def test_support_and_candidate_order_invariant():
    base, phi, env, support, labels = synthetic()
    first, _ = adapt_predict(base, phi, env, support, labels[support])
    reordered, _ = adapt_predict(base, phi, env, support.flip(0), labels[support.flip(0)])
    torch.testing.assert_close(first, reordered, rtol=1e-12, atol=1e-12)
    permutation = torch.randperm(len(base), generator=torch.Generator().manual_seed(31))
    inverse = torch.argsort(permutation)
    permuted, _ = adapt_predict(base[permutation], phi[permutation], env[permutation], inverse[support], labels[support])
    torch.testing.assert_close(first, permuted[inverse], rtol=1e-12, atol=1e-12)


def test_intercept_only_preserves_all_within_environment_differences():
    base, phi, env, support, labels = synthetic()
    prediction, diag = adapt_predict(base, phi, env, support, labels[support], mode="intercept")
    for e in torch.unique(env):
        indices = torch.where(env == e)[0]
        torch.testing.assert_close(prediction[indices] - prediction[indices[0]], base[indices] - base[indices[0]])
    assert torch.count_nonzero(diag["within_environment_update_rms"]) == 0


def test_hidden_labels_not_consumed_and_s0_not_duplicated():
    base, phi, env, support, labels = synthetic()
    hidden = torch.ones(len(base), dtype=torch.bool)
    hidden[support] = False
    changed = labels.clone()
    changed[hidden] = 1e8
    first, _ = adapt_predict(base, phi, env, support, labels[support])
    second, _ = adapt_predict(base, phi, env, support, changed[support])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    with pytest.raises(ValueError, match="S0"):
        adapt_predict(base, phi, env, torch.cat([support, support[:1]]), torch.cat([labels[support], labels[support[:1]]]))
    with pytest.raises(ValueError, match="支持标签"):
        adapt_predict(base, phi, env, support, labels)


def test_no_support_is_exact_base():
    base, phi, env, _, _ = synthetic()
    empty = torch.empty(0, dtype=torch.long)
    for mode in ("zero", "response", "intercept"):
        prediction, diag = adapt_predict(base, phi, env, empty, torch.empty(0), mode=mode)
        torch.testing.assert_close(prediction, base, rtol=0, atol=0)
        assert diag["support_count"] == 0
    with pytest.raises(ValueError, match="zero"):
        adapt_predict(base, phi, env, torch.tensor([0]), torch.zeros(1), mode="zero")


def test_gradients_through_joint_solve_and_whole_pool_centering():
    base, phi, env, support, labels = synthetic()
    base.requires_grad_()
    phi.requires_grad_()
    prediction, _ = adapt_predict(base, phi, env, support, labels[support])
    # 仅一行目标损失；同环境未观测且不在损失里的行仍经 V 中心化收到梯度。
    loss = (prediction[2] - labels[2]).square().mean()
    loss.backward()
    assert torch.isfinite(base.grad).all() and torch.isfinite(phi.grad).all()
    assert base.grad[support].abs().sum() > 0
    assert phi.grad[5].abs().sum() > 0
    def scalar(p):
        y, _ = adapt_predict(base.detach(), p, env, support, labels[support])
        return y[2].sum()
    delta = 1e-5
    plus, minus = phi.detach().clone(), phi.detach().clone()
    plus[5, 0, 0] += delta
    minus[5, 0, 0] -= delta
    numeric = (scalar(plus) - scalar(minus)) / (2 * delta)
    analytic = torch.autograd.grad(scalar(phi), phi)[0][5, 0, 0]
    torch.testing.assert_close(analytic, numeric, rtol=1e-5, atol=1e-8)


def test_official_tabm_equivalence_projection_and_state_reproducible():
    model = ResponseTabM(5, seed=1009)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1009)
        official = TabM.make(n_num_features=5, d_out=1, num_embeddings=None, n_blocks=3,
                            d_block=128, k=16, dropout=0., activation="ReLU",
                            arch_type="tabm", start_scaling_init="random-signs")
    x = torch.randn(11, 5)
    prediction, phi = model(x)
    torch.testing.assert_close(prediction, official(x).squeeze(-1), rtol=0, atol=0)
    assert prediction.shape == (11, 16) and phi.shape == (11, 16, 8)
    torch.testing.assert_close(model.projection.T @ model.projection, torch.eye(8), atol=2e-7, rtol=1e-6)
    other = ResponseTabM(5, seed=2027)
    assert model.projection_sha256 == other.projection_sha256
    other.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(model(x)[0], other(x)[0], rtol=0, atol=0)


def test_chunking_preserves_outputs_and_parameter_gradients():
    model = ResponseTabM(5, seed=3037)
    replica = copy.deepcopy(model)
    x = torch.randn(12, 5)
    base, phi = model(x)
    chunk_base, chunk_phi = replica.forward_chunked(x, chunk_size=4)
    torch.testing.assert_close(base, chunk_base, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(phi, chunk_phi, rtol=1e-5, atol=1e-6)
    env = torch.tensor([0] * 6 + [1] * 6)
    support = torch.tensor([0, 1, 6, 7])
    labels = torch.tensor([1., -1., .5, 2.])
    for b, p in ((base, phi), (chunk_base, chunk_phi)):
        pred, _ = adapt_predict(b, p, env, support, labels)
        pred[torch.tensor([3, 9])].square().mean().backward()
    for p, q in zip(model.parameters(), replica.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6)


def test_pairwise_square_loss_is_centered_square_loss_identity():
    error = np.random.default_rng(41).normal(size=13)
    all_unordered_pairs = sum((error[i] - error[j]) ** 2 for i in range(len(error)) for j in range(i + 1, len(error)))
    np.testing.assert_allclose(all_unordered_pairs, len(error) * np.square(error - error.mean()).sum())
