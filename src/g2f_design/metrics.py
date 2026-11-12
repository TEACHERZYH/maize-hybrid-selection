"""观测筛选指标；保留未支持、无收益和缺失状态。"""
import math
import numpy as np
from scipy.stats import spearmanr


def ranking_metrics(observed, predicted, identifiers):
    y, p = np.asarray(observed, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if not len(y) or len(y) != len(p) or len(y) != len(identifiers) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('排名数据缺失或形状错误')
    k = math.ceil(0.2 * len(y))
    truth = sorted(range(len(y)), key=lambda i: (-y[i], identifiers[i]))[:k]
    chosen = sorted(range(len(y)), key=lambda i: (-p[i], identifiers[i]))[:k]
    regret = float(np.mean(y[truth]) - np.mean(y[chosen]))
    if regret < -1e-10:
        raise ValueError('观测选择遗憾为实质负值')
    correlation = None if np.ptp(y) == 0 or np.ptp(p) == 0 else float(spearmanr(y, p).statistic)
    return {'retention': len(set(truth).intersection(chosen)) / k, 'regret': max(regret, 0.0),
            'rmse': float(np.sqrt(np.mean((y - p) ** 2))), 'spearman': correlation, 'k': k}


def regret_auc(regrets, integer_budgets, query_pool_size, s_ref):
    if len(regrets) != 4 or len(integer_budgets) != 4 or s_ref <= 1e-12:
        raise ValueError('四点主曲线不完整或历史尺度无效')
    y = np.asarray(regrets, dtype=np.float64) / s_ref
    x = np.asarray(integer_budgets, dtype=np.float64) / query_pool_size
    if not np.isfinite(y).all() or not np.all(np.diff(x) > 0):
        raise ValueError('预算或指标无效')
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2) / (x[-1] - x[0]))


def observed_savings(integer_budgets, query_pool_size, retention, standardized_regret,
                     full_retention, full_standardized_regret):
    budgets = np.asarray(integer_budgets, dtype=int)
    if len(budgets) != 5 or budgets[-1] != query_pool_size or not np.all(np.diff(budgets) > 0):
        raise ValueError('节省计算需要完整五点含100%参照')
    keep = ((np.asarray(retention) >= full_retention - 0.05)
            & (np.asarray(standardized_regret) <= full_standardized_regret + 0.1))
    eligible = [i for i in range(len(budgets)) if keep[i:].all()]
    if not eligible:
        raise ValueError('100%参照不满足自身阈值，记录不一致')
    minimum = int(budgets[eligible[0]])
    return {'minimum_observations': minimum, 'savings': 1 - minimum / query_pool_size}
