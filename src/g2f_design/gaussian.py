"""双精度有限特征高斯模型；训练只使用充分统计量。"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

NUMERICAL_EVENTS = []


def factor(matrix):
    a = np.asarray(matrix, dtype=np.float64)
    scale = max(1.0, float(np.max(np.diag(a))))
    if not np.allclose(a, a.T, rtol=0, atol=1e-12 * scale):
        raise np.linalg.LinAlgError('矩阵不对称')
    a = (a + a.T) / 2
    try:
        return cho_factor(a, lower=True, check_finite=True)
    except np.linalg.LinAlgError:
        if np.linalg.eigvalsh(a)[0] < -1e-12 * scale:
            raise np.linalg.LinAlgError('出现实质负特征值')
        for multiplier in (1e-12, 1e-10):
            try:
                result = cho_factor(a + np.eye(len(a)) * multiplier * scale, lower=True)
                NUMERICAL_EVENTS.append({'operation': 'cholesky_jitter', 'dimension': len(a),
                                         'scale': scale, 'multiplier': multiplier})
                return result
            except np.linalg.LinAlgError:
                pass
        raise np.linalg.LinAlgError('允许的数值抖动包络耗尽')


@dataclass
class Posterior:
    mean: np.ndarray
    covariance: np.ndarray
    noise: float

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=np.float64).copy()
        self.covariance = np.asarray(self.covariance, dtype=np.float64).copy()
        self.noise = float(self.noise)
        if self.covariance.shape != (len(self.mean), len(self.mean)) or self.noise <= 0:
            raise ValueError('后验维度或独立噪声非法')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.covariance).all() or not np.isfinite(self.noise):
            raise ValueError('后验含非有限值')

    def copy(self):
        return Posterior(self.mean.copy(), self.covariance.copy(), self.noise)

    def condition_position(self, row):
        r = self.covariance @ row
        variance = float(row @ r + self.noise)
        if variance <= 0:
            raise np.linalg.LinAlgError('观测方差非正')
        self.covariance -= np.outer(r, r) / variance
        self.covariance = (self.covariance + self.covariance.T) / 2

    def condition_labels(self, rows, values):
        a = np.asarray(rows, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if not len(a):
            return self.copy()
        if len(a) > len(self.mean):
            precision = cho_solve(factor(self.covariance), np.eye(len(self.mean)))
            updated = factor(precision + a.T @ a / self.noise)
            covariance = cho_solve(updated, np.eye(len(self.mean)))
            mean = self.mean + cho_solve(updated, a.T @ (values - a @ self.mean) / self.noise)
            return Posterior(mean, (covariance + covariance.T) / 2, self.noise)
        r = self.covariance @ a.T
        system = a @ r + np.eye(len(a)) * self.noise
        chol = factor(system)
        mean = self.mean + r @ cho_solve(chol, values - a @ self.mean)
        covariance = self.covariance - r @ cho_solve(chol, r.T)
        return Posterior(mean, (covariance + covariance.T) / 2, self.noise)


@dataclass
class SufficientStatistics:
    n: int
    xx: np.ndarray
    xy: np.ndarray
    yy: float

    @classmethod
    def from_arrays(cls, x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        return cls(len(y), x.T @ x, x.T @ y, float(y @ y))

    def add(self, x, y):
        block = self.from_arrays(x, y)
        self.n += block.n
        self.xx += block.xx
        self.xy += block.xy
        self.yy += block.yy


def likelihood(log_parameters, statistics, groups):
    """groups中的-1代表固定方差100的全局截距，其余为随机块编号。"""
    parameters = np.exp(log_parameters)
    noise = float(parameters[-1])
    groups = np.asarray(groups, dtype=int)
    prior = np.full(len(groups), 100.0)
    random = groups >= 0
    prior[random] = parameters[groups[random]]
    precision = np.diag(1 / prior) + statistics.xx / noise
    chol = factor(precision)
    covariance = cho_solve(chol, np.eye(len(groups)))
    mean = cho_solve(chol, statistics.xy / noise)
    logdet = statistics.n * np.log(noise) + np.log(prior).sum() + 2 * np.log(np.diag(chol[0])).sum()
    quadratic = statistics.yy / noise - (statistics.xy / noise) @ mean
    nll = 0.5 * (statistics.n * np.log(2 * np.pi) + logdet + quadratic)
    second = np.diag(covariance) + mean ** 2
    gradient = []
    for group in range(len(parameters) - 1):
        mask = groups == group
        gradient.append(0.5 * (mask.sum() - second[mask].sum() / parameters[group]))
    expected_residual = (statistics.yy - 2 * mean @ statistics.xy + mean @ statistics.xx @ mean
                         + np.einsum('ij,ji->', statistics.xx, covariance))
    gradient.append(0.5 * (statistics.n - expected_residual / noise))
    return float(nll), np.asarray(gradient), Posterior(mean, covariance, noise)


def fit_variances(statistics, groups):
    groups = np.asarray(groups, dtype=int)
    count = int(groups.max()) + 1
    if sorted(set(groups[groups >= 0])) != list(range(count)):
        raise ValueError('方差块编号不连续')
    results = []
    for name, block, noise in [('A', 0.1, 0.5), ('B', 0.5, 0.1), ('C', 0.01, 1.0)]:
        initial = np.log([block] * count + [noise])
        result = minimize(lambda p: likelihood(p, statistics, groups)[:2], initial,
                          method='L-BFGS-B', jac=True, bounds=[(np.log(1e-8), np.log(100))] * len(initial),
                          options={'maxiter': 500, 'maxfun': 20000, 'maxls': 40, 'ftol': 1e-9, 'gtol': 1e-6})
        results.append({'start': name, 'success': bool(result.success), 'nll': float(result.fun),
                        'iterations': int(result.nit), 'message': str(result.message), 'log_parameters': result.x.tolist()})
    valid = [r for r in results if r['success'] and np.isfinite(r['nll'])]
    if not valid:
        raise RuntimeError('三组已定方差初值均未收敛：' + str(results))
    best_value = min(r['nll'] for r in valid)
    best = next(r for r in valid if r['nll'] <= best_value + 1e-10)
    nll, gradient, posterior = likelihood(np.asarray(best['log_parameters']), statistics, groups)
    return posterior, {'chosen_start': best['start'], 'starts': results, 'nll': nll,
                       'gradient_max_abs': float(np.max(np.abs(gradient))),
                       'variances': np.exp(best['log_parameters']).tolist()}
