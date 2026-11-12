"""精确积分组合共享差异及物理区组，不构造历史小区方阵。"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from .gaussian import Posterior, factor


@dataclass
class EnvironmentStatistics:
    x: np.ndarray
    y: np.ndarray
    ci: np.ndarray
    bi: object
    nc: np.ndarray
    nb: np.ndarray
    incidence: np.ndarray
    sy: np.ndarray

    @classmethod
    def from_arrays(cls, x, y, combo_index, block_index=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ci = np.asarray(combo_index, int)
        if (x.ndim != 2 or y.ndim != 1 or len(ci) != len(y) or not len(y)
                or set(ci) != set(range(len(x))) or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError('组合统计量身份或数值无效')
        bi = None if block_index is None else np.asarray(block_index, int)
        if bi is not None and (len(bi) != len(y) or set(bi) != set(range(int(bi.max()) + 1))):
            raise ValueError('环境内区组索引不连续')
        nc = np.bincount(ci, minlength=len(x)).astype(float)
        nb = np.zeros(0) if bi is None else np.bincount(bi).astype(float)
        incidence = np.zeros((len(x), len(nb)))
        if bi is not None:
            np.add.at(incidence, (ci, bi), 1)
        return cls(x, y, ci, bi, nc, nb, incidence, np.bincount(ci, weights=y, minlength=len(x)))


def _environment_precision(stat, combo, block, noise):
    x, y, ci, bi, nc, nb, inc, sy = (stat.x, stat.y, stat.ci, stat.bi, stat.nc,
                                    stat.nb, stat.incidence, stat.sy)
    q = 1 / (noise + nc * combo)
    w = combo * q / noise
    rx = q[ci, None] * x[ci]
    ry = y / noise - w[ci] * sy[ci]
    logdet = len(y) * np.log(noise) + np.log1p(nc * combo / noise).sum()
    trace_r = len(y) / noise - nc @ w
    trace_c = nc @ q
    trace_b = 0.0
    if bi is not None:
        dinvz = -w[ci, None] * inc[ci]
        dinvz[np.arange(len(y)), bi] += 1 / noise
        m = np.diag(nb) / noise - inc.T @ (w[:, None] * inc)
        chol = factor(np.eye(len(nb)) / block + m)
        invb = cho_solve(chol, np.eye(len(nb)))
        zx = inc.T @ (q[:, None] * x)
        zy = np.bincount(bi, weights=ry, minlength=len(nb))
        rx -= dinvz @ (invb @ zx)
        ry -= dinvz @ (invb @ zy)
        logdet += len(nb) * np.log(block) + 2 * np.log(np.diag(chol[0])).sum()
        dz2 = np.diag(nb) / noise ** 2 - inc.T @ ((2*w/noise-nc*w*w)[:, None] * inc)
        trace_r -= np.einsum('ij,ji->', invb, dz2)
        trace_c -= (q*q) @ np.einsum('ij,ij->i', inc @ invb, inc)
        trace_b = np.trace(m) - np.einsum('ij,ji->', invb, m @ m)
    return rx, ry, float(logdet), np.asarray([trace_c, trace_b, trace_r])


def crossed_likelihood(log_parameters, statistics, groups):
    groups = np.asarray(groups, int)
    core_count = int(groups.max()) + 1
    has_block = statistics[0].bi is not None
    parameters = np.exp(log_parameters)
    if (len(parameters) != core_count + 2 + int(has_block)
            or any((s.bi is not None) != has_block for s in statistics)):
        raise ValueError('交叉随机效应参数或区组开关不一致')
    combo = float(parameters[core_count])
    block = float(parameters[-2]) if has_block else None
    noise = float(parameters[-1])
    prior = np.full(len(groups), 100.0)
    mask = groups >= 0
    prior[mask] = parameters[groups[mask]]
    precision = np.diag(1 / prior)
    rhs = np.zeros(len(groups))
    logdet, yy, count = 0.0, 0.0, 0
    terms = []
    for stat in statistics:
        rx, ry, ld, traces = _environment_precision(stat, combo, block, noise)
        x = stat.x[stat.ci]
        precision += x.T @ rx
        rhs += x.T @ ry
        yy += stat.y @ ry
        logdet += ld
        count += len(stat.y)
        terms.append((stat, rx, ry, traces))
    precision = (precision + precision.T) * .5
    chol = factor(precision)
    covariance = cho_solve(chol, np.eye(len(groups)))
    mean = cho_solve(chol, rhs)
    logdet += np.log(prior).sum() + 2 * np.log(np.diag(chol[0])).sum()
    nll = .5 * (count*np.log(2*np.pi) + logdet + yy - rhs @ mean)
    second = np.diag(covariance) + mean*mean
    gradient = [.5*(np.sum(groups == g)-second[groups == g].sum()/parameters[g]) for g in range(core_count)]
    trace_sums = np.zeros(3)
    square_sums = np.zeros(3)
    products = [np.zeros_like(covariance) for _ in range(3)]
    for stat, rx, ry, traces in terms:
        a = ry - rx @ mean
        tc = np.zeros_like(stat.x)
        np.add.at(tc, stat.ci, rx)
        ac = np.bincount(stat.ci, weights=a, minlength=len(tc))
        products[0] += tc.T @ tc
        products[2] += rx.T @ rx
        square_sums += [ac @ ac, 0.0, a @ a]
        if has_block:
            tb = np.zeros((len(stat.nb), len(groups)))
            np.add.at(tb, stat.bi, rx)
            ab = np.bincount(stat.bi, weights=a, minlength=len(tb))
            products[1] += tb.T @ tb
            square_sums[1] += ab @ ab
        trace_sums += traces
    residual_gradient = .5*(trace_sums - square_sums - np.array([
        np.einsum('ij,ji->', p, covariance) for p in products]))
    gradient.append(combo * residual_gradient[0])
    if has_block:
        gradient.append(block * residual_gradient[1])
    gradient.append(noise * residual_gradient[2])
    return float(nll), np.asarray(gradient), Posterior(mean, covariance, noise)


def fit_crossed_variances(statistics, groups):
    groups = np.asarray(groups, int)
    count = int(groups.max()) + 1
    if not statistics or sorted(set(groups[groups >= 0])) != list(range(count)):
        raise ValueError('历史统计量缺失或方差块编号不连续')
    has_block = statistics[0].bi is not None
    results = []
    for name, random, noise in [('A', .1, .5), ('B', .5, .1), ('C', .01, 1.0)]:
        initial = np.log([random]*(count + 1 + int(has_block)) + [noise])
        result = minimize(lambda p: crossed_likelihood(p, statistics, groups)[:2], initial,
                          method='L-BFGS-B', jac=True,
                          bounds=[(np.log(1e-8), np.log(100))]*len(initial),
                          options={'maxiter': 500, 'maxfun': 20000, 'maxls': 40, 'ftol': 1e-9, 'gtol': 1e-6})
        results.append({'start': name, 'success': bool(result.success), 'nll': float(result.fun),
                        'iterations': int(result.nit), 'message': str(result.message),
                        'log_parameters': result.x.tolist()})
    valid = [r for r in results if r['success'] and np.isfinite(r['nll'])]
    if not valid:
        raise RuntimeError('三组冻结交叉方差初值均未收敛：'+str(results))
    best = next(r for r in valid if r['nll'] <= min(x['nll'] for x in valid)+1e-10)
    nll, gradient, posterior = crossed_likelihood(best['log_parameters'], statistics, groups)
    return posterior, {'chosen_start': best['start'], 'starts': results, 'nll': nll,
                       'gradient_max_abs': float(np.abs(gradient).max()),
                       'variances': np.exp(best['log_parameters']).tolist(),
                       'history_elimination': 'exact_combo_and_physical_block',
                       'historical_combinations': sum(len(s.x) for s in statistics),
                       'historical_blocks': sum(len(s.nb) for s in statistics)}
