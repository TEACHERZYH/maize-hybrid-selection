"""固定查询集合后的潜在排序与真实复验小区预测。"""
import numpy as np
from scipy.linalg import cho_solve
from .gaussian import factor


def predict_replicates(posterior, x, selected, values, query_blocks, audit_blocks,
                       combo_variance=0.0, block_variance=0.0):
    x, selected, values = np.asarray(x, float), np.asarray(selected, int), np.asarray(values, float)
    if (len(selected) != len(values) or len(set(selected)) != len(selected)
            or np.any(selected < 0) or np.any(selected >= len(x))
            or combo_variance < 0 or block_variance < 0 or not np.isfinite(values).all()):
        raise ValueError('查询身份或共享方差无效')
    mean = x @ posterior.mean
    cross = x @ posterior.covariance @ x[selected].T
    cross[selected, np.arange(len(selected))] += combo_variance
    kqq = cross[selected].copy()
    kaq = cross.copy()
    if block_variance:
        q, a = np.asarray(query_blocks), np.asarray(audit_blocks)
        if len(q) != len(x) or len(a) != len(x):
            raise ValueError('真实小区物理区组不完整')
        kqq += block_variance * (q[selected, None] == q[None, selected])
        kaq += block_variance * (a[:, None] == q[None, selected])
    kqq[np.diag_indices(len(selected))] += posterior.noise
    chol = factor((kqq+kqq.T)*.5)
    alpha = cho_solve(chol, values-mean[selected])
    latent_mean = mean + cross @ alpha
    observation_mean = mean + kaq @ alpha
    variance = (np.einsum('ij,ij->i', x @ posterior.covariance, x) + combo_variance
                + block_variance + posterior.noise - np.einsum('ij,ji->i', kaq, cho_solve(chol, kaq.T)))
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError('审计观测预测方差非正或非有限')
    return latent_mean, observation_mean, variance


def calibration_metrics(observed, mean, variance, scale):
    y, mu, v = np.asarray(observed, float), np.asarray(mean, float), np.asarray(variance, float)
    if len(y) != len(mu) or len(y) != len(v) or np.any(v <= 0) or not np.isfinite(y).all():
        raise ValueError('校准数据无效')
    radius = 1.6448536269514722 * np.sqrt(v)
    return {'nlpd': float(np.mean(.5*(np.log(2*np.pi*v)+(y-mu)**2/v))),
            'coverage90': float(np.mean(np.abs(y-mu) <= radius)),
            'width90_standardized': float(np.mean(2*radius)),
            'width90_original_units': float(np.mean(2*radius)*scale)}
