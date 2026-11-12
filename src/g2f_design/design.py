"""冻结均值、权重的精确H目标与批次选择，不接受当季产量输入。"""
import math
import numpy as np
from scipy.special import log_ndtr, logsumexp


class DegenerateDesign(RuntimeError):
    pass


def best_index(scores, identifiers, legal):
    indices = np.flatnonzero(legal)
    if not len(indices):
        raise ValueError('没有合法候选')
    values = np.asarray(scores)[indices]
    if np.isnan(values).any() or np.isposinf(values).any():
        raise FloatingPointError('分数出现非有限异常')
    maximum = float(np.max(values))
    if maximum == -np.inf:
        return min(indices, key=lambda i: identifiers[i])
    tolerance = 1e-12 * max(1.0, abs(maximum))
    tied = indices[values >= maximum - tolerance]
    return min(tied, key=lambda i: identifiers[i])


def objective_h(posterior, targets, identifiers, environments, kind='M', variance_floor=1e-8):
    """targets只含潜在f的系数；区组观察项必须为0。"""
    targets = np.asarray(targets, dtype=np.float64)
    environments = np.asarray(environments)
    unique = sorted(set(environments))
    matrix = np.zeros((targets.shape[1], targets.shape[1]))
    means = targets @ posterior.mean
    active = 0
    diagnostics = []
    for env in unique:
        indices = np.flatnonzero(environments == env)
        if kind == 'B3':
            contrasts = targets[indices] - targets[indices].mean(axis=0)
            matrix += contrasts.T @ contrasts / (len(unique) * len(indices))
            continue
        order = sorted(indices, key=lambda i: (-means[i], identifiers[i]))
        k = math.ceil(0.2 * len(order))
        group = targets[order[:k]]
        rest = targets[order[k:]]
        contrasts = (group[:, None, :] - rest[None, :, :]).reshape(-1, targets.shape[1])
        if not len(contrasts):
            diagnostics.append({'env': str(env), 'valid_pairs': 0})
            continue
        variances = np.einsum('ij,ij->i', contrasts @ posterior.covariance, contrasts)
        keep = variances > variance_floor
        contrasts, variances = contrasts[keep], variances[keep]
        diagnostics.append({'env': str(env), 'valid_pairs': len(variances)})
        if not len(contrasts):
            continue
        if kind == 'M':
            z = contrasts @ posterior.mean / np.sqrt(variances)
            log_weights = np.log(4) + log_ndtr(z) + log_ndtr(-z)
            weights = np.exp(log_weights - logsumexp(log_weights))
        elif kind == 'B4':
            weights = np.full(len(variances), 1 / len(variances))
        else:
            raise ValueError('未知H目标')
        matrix += (contrasts.T * (weights / variances)) @ contrasts / len(unique)
        active += len(variances)
    if kind != 'B3' and not active:
        raise DegenerateDesign('所有环境都没有有效品种对，停止该机制分支')
    return (matrix + matrix.T) / 2, diagnostics


def marginal_scores(covariance, observations, noise, h):
    r = observations @ covariance
    denominator = np.einsum('ij,ij->i', r, observations) + noise
    if np.any(denominator <= 0):
        raise FloatingPointError('候选观测方差非正')
    numerator = np.einsum('ij,ij->i', r @ h, r)
    scale = max(1.0, float(np.max(np.abs(numerator))))
    if np.min(numerator) < -1e-10 * scale:
        raise FloatingPointError('H边际出现实质负值')
    return np.maximum(numerator, 0) / denominator


def greedy_select(posterior, observations, identifiers, budget, initial, h=None, kind='M', latent=None,
                  prescreen=None):
    observations = np.asarray(observations, dtype=np.float64)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('候选ID重复')
    selected = list(initial)
    if len(set(selected)) != len(selected) or not len(selected) <= budget <= len(observations):
        raise ValueError('必选集或预算不合法')
    legal = np.ones(len(observations), dtype=bool)
    state = posterior.copy()
    for index in selected:
        if index < 0 or index >= len(observations):
            raise ValueError('必选对象不在查询池')
        legal[index] = False
        state.condition_position(observations[index])
    fixed_scores = None
    if kind == 'M-static':
        fixed_scores = marginal_scores(state.covariance, observations, state.noise, h)
    for _ in range(budget - len(selected)):
        available = legal.copy()
        if prescreen is not None:
            suggestion = prescreen.suggest(np.flatnonzero(legal))
            available[:] = False
            available[suggestion] = True
            if np.any(available & ~legal):
                raise ValueError('预筛建议包含非法对象')
        indices = np.flatnonzero(available)
        scores = np.full(len(observations), -np.inf)
        if kind == 'B5':
            if latent is None:
                raise ValueError('不确定性基线必须提供潜在f的特征')
            scores[indices] = np.einsum('ij,ij->i', latent[indices] @ state.covariance, latent[indices])
        else:
            scores[indices] = (fixed_scores[indices] if fixed_scores is not None else
                               marginal_scores(state.covariance, observations[indices], state.noise, h))
        chosen = best_index(scores, identifiers, available)
        selected.append(chosen)
        legal[chosen] = False
        state.condition_position(observations[chosen])
        if prescreen is not None:
            prescreen.add(chosen)
    return selected, state


class LabelOracle:
    """回放中的查询闸门；先锁整批，之后才揭示允许查询的标签。"""
    def __init__(self, query_values, audit_ids):
        self._values = dict(query_values)
        self._audit = set(audit_ids)
        if self._audit.intersection(self._values):
            raise ValueError('查询池与审计池重叠')
        self._locked = None
        self.revealed_count = 0

    def lock(self, identifiers, expected_cost):
        if self._locked is not None or len(identifiers) != expected_cost:
            raise ValueError('批次重复锁定或成本不符')
        if len(set(identifiers)) != len(identifiers) or any(i not in self._values for i in identifiers):
            raise ValueError('批次有重复、审计或未知对象')
        self._locked = tuple(identifiers)

    def reveal(self):
        if self._locked is None:
            raise PermissionError('整批未锁定，禁止读取当季真实标签')
        self.revealed_count = len(self._locked)
        return np.asarray([self._values[i] for i in self._locked], dtype=np.float64)
