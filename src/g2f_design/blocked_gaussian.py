"""精确消去历史独立区组；保留共同模型与冻结方差拟合条件。"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from .gaussian import SufficientStatistics, Posterior, factor


@dataclass
class BlockedStatistics:
    core: SufficientStatistics
    counts: np.ndarray
    sums_x: np.ndarray
    sums_y: np.ndarray

    @classmethod
    def empty(cls, dimension, blocks):
        core = SufficientStatistics(0, np.zeros((dimension, dimension)), np.zeros(dimension), 0.0)
        return cls(core, np.zeros(blocks), np.zeros((blocks, dimension)), np.zeros(blocks))

    def add(self, x, y, block_indices):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        indices = np.asarray(block_indices, dtype=int)
        if len(indices) != len(y) or np.any(indices < 0) or np.any(indices >= len(self.counts)):
            raise ValueError('区组索引不合法')
        self.core.add(x, y)
        np.add.at(self.counts, indices, 1)
        np.add.at(self.sums_x, indices, x)
        np.add.at(self.sums_y, indices, y)


def blocked_likelihood(log_parameters, statistics, groups):
    groups = np.asarray(groups, dtype=int)
    parameters = np.exp(log_parameters)
    random_count = int(groups.max()) + 1
    if len(parameters) != random_count + 2 or np.any(statistics.counts <= 0):
        raise ValueError('区组方差参数或充分统计量不完整')
    variance, noise = map(float, parameters[-2:])
    prior = np.full(len(groups), 100.0)
    mask = groups >= 0
    prior[mask] = parameters[groups[mask]]
    core, n = statistics.core, statistics.counts
    v, sums_y = statistics.sums_x, statistics.sums_y
    a = variance / (noise + n * variance)
    weight = a / noise
    precision = np.diag(1 / prior) + core.xx / noise - v.T @ (weight[:, None] * v)
    rhs = core.xy / noise - v.T @ (weight * sums_y)
    chol = factor(precision)
    covariance = cho_solve(chol, np.eye(len(groups)))
    mean = cho_solve(chol, rhs)
    logdet = (core.n * np.log(noise) + np.log1p(n * variance / noise).sum()
              + np.log(prior).sum() + 2 * np.log(np.diag(chol[0])).sum())
    quadratic = core.yy / noise - np.dot(weight, sums_y ** 2) - rhs @ mean
    nll = .5 * (core.n * np.log(2 * np.pi) + logdet + quadratic)
    second = np.diag(covariance) + mean ** 2
    gradient = [.5 * (np.sum(groups == g) - np.sum(second[groups == g]) / parameters[g])
                for g in range(random_count)]
    projected = np.einsum('ij,ij->i', v @ covariance, v)
    residual_sums = sums_y - v @ mean
    block_mean = a * residual_sums
    block_variance = noise * a + a ** 2 * projected
    gradient.append(.5 * (len(n) - np.sum(block_mean ** 2 + block_variance) / variance))
    residual = (core.yy - 2 * mean @ core.xy + mean @ core.xx @ mean
                - 2 * block_mean @ residual_sums + n @ (block_mean ** 2 + block_variance)
                + np.einsum('ij,ji->', core.xx, covariance) - 2 * a @ projected)
    gradient.append(.5 * (core.n - residual / noise))
    return float(nll), np.asarray(gradient), Posterior(mean, covariance, noise)


def fit_blocked_variances(statistics, groups):
    groups = np.asarray(groups, dtype=int)
    count = int(groups.max()) + 1
    if sorted(set(groups[groups >= 0])) != list(range(count)):
        raise ValueError('核心方差块编号不连续')
    results = []
    for name, block, noise in [('A', .1, .5), ('B', .5, .1), ('C', .01, 1.0)]:
        initial = np.log([block] * (count + 1) + [noise])
        result = minimize(lambda p: blocked_likelihood(p, statistics, groups)[:2], initial,
                          method='L-BFGS-B', jac=True,
                          bounds=[(np.log(1e-8), np.log(100))] * len(initial),
                          options={'maxiter': 500, 'maxfun': 20000, 'maxls': 40, 'ftol': 1e-9, 'gtol': 1e-6})
        results.append({'start': name, 'success': bool(result.success), 'nll': float(result.fun),
                        'iterations': int(result.nit), 'message': str(result.message),
                        'log_parameters': result.x.tolist()})
    valid = [r for r in results if r['success'] and np.isfinite(r['nll'])]
    if not valid:
        raise RuntimeError('三组冻结区组方差初值均未收敛：' + str(results))
    best = next(r for r in valid if r['nll'] <= min(x['nll'] for x in valid) + 1e-10)
    nll, gradient, posterior = blocked_likelihood(np.asarray(best['log_parameters']), statistics, groups)
    return posterior, {'chosen_start': best['start'], 'starts': results, 'nll': nll,
                       'gradient_max_abs': float(np.max(np.abs(gradient))),
                       'variances': np.exp(best['log_parameters']).tolist(),
                       'history_block_elimination': 'exact', 'historical_blocks': len(statistics.counts)}


def append_independent_blocks(posterior, block_count, variance):
    if block_count < 0 or variance <= 0:
        raise ValueError('当期独立区组先验非法')
    d = len(posterior.mean)
    covariance = np.zeros((d + block_count, d + block_count))
    covariance[:d, :d] = posterior.covariance
    covariance[d:, d:] = np.eye(block_count) * variance
    return Posterior(np.r_[posterior.mean, np.zeros(block_count)], covariance, posterior.noise)
