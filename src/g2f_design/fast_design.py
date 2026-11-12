"""精确秩一更新缓存；与原稠密H评分相同，不作候选近似。"""
import numpy as np
from .design import best_index, marginal_scores


class ExactGreedy:
    def __init__(self, posterior, observations, identifiers, initial, h=None, kind='M',
                 latent=None, resume=None, refresh_every=256):
        self.a = np.asarray(observations, dtype=np.float64)
        self.ids, self.h, self.kind = list(identifiers), h, kind
        self.latent = None if latent is None else np.asarray(latent, dtype=np.float64)
        self.initial = list(initial)
        self.selected = list(initial if resume is None else resume)
        if (len(set(self.ids)) != len(self.ids) or len(self.a) != len(self.ids)
                or len(set(self.selected)) != len(self.selected)
                or self.selected[:len(initial)] != list(initial)
                or any(i < 0 or i >= len(self.a) for i in self.selected)):
            raise ValueError('候选、必选或恢复序列不合法')
        if kind not in ('M', 'B3', 'B4', 'M-static', 'B5'):
            raise ValueError('未知精确设计方法')
        if kind == 'B5' and (self.latent is None or self.latent.shape != self.a.shape):
            raise ValueError('B5需要与观测行对应的潜在表现特征')
        if kind != 'B5' and (h is None or np.shape(h) != (self.a.shape[1], self.a.shape[1])):
            raise ValueError('H维度不合法')
        self.state = posterior.copy()
        self.legal = np.ones(len(self.a), dtype=bool)
        self.legal[self.selected] = False
        self.refresh_every = int(refresh_every)
        if self.refresh_every < 1:
            raise ValueError('数值刷新间隔须为正')
        for i in self.initial:
            self.state.condition_position(self.a[i])
        self.fixed = (marginal_scores(self.state.covariance, self.a, self.state.noise, h)
                      if kind == 'M-static' else None)
        for i in self.selected[len(self.initial):]:
            self.state.condition_position(self.a[i])
        self.refresh()

    def refresh(self):
        if self.kind == 'M-static':
            return
        if self.kind == 'B5':
            self.variance = np.einsum('ij,ij->i', self.latent @ self.state.covariance, self.latent)
        else:
            self.r = self.a @ self.state.covariance
            self.denominator = np.einsum('ij,ij->i', self.r, self.a) + self.state.noise
            self.numerator = np.einsum('ij,ij->i', self.r @ self.h, self.r)

    def scores(self):
        if self.kind == 'M-static':
            return self.fixed
        if self.kind == 'B5':
            scores = self.variance
        else:
            scale = max(1.0, float(np.max(np.abs(self.numerator))))
            if np.min(self.denominator) <= 0 or np.min(self.numerator) < -1e-10 * scale:
                self.refresh()
            if np.min(self.denominator) <= 0 or np.min(self.numerator) < -1e-10 * scale:
                raise FloatingPointError('精确缓存刷新后仍出现非法方差或H边际')
            scores = np.maximum(self.numerator, 0) / self.denominator
        if not np.isfinite(scores).all():
            raise FloatingPointError('精确设计分数含非有限值')
        return scores

    def step(self):
        chosen = best_index(self.scores(), self.ids, self.legal)
        row = self.a[chosen]
        r = self.state.covariance @ row
        variance = float(row @ r + self.state.noise)
        if variance <= 0:
            raise FloatingPointError('观测方差非正')
        if self.kind == 'B5':
            self.variance -= (self.latent @ r) ** 2 / variance
        elif self.kind != 'M-static':
            c = self.a @ r
            hr = self.h @ r
            self.numerator -= 2 * c * (self.r @ hr) / variance
            self.numerator += c ** 2 * float(r @ hr) / variance ** 2
            self.denominator -= c ** 2 / variance
            self.r -= np.outer(c, r) / variance
        # 与原稠密实现按相同位置顺序执行条件化，始终不接收标签。
        self.state.condition_position(row)
        self.selected.append(chosen)
        self.legal[chosen] = False
        if (len(self.selected) - len(self.initial)) % self.refresh_every == 0:
            self.refresh()
        return chosen

    def take(self, budget):
        if not len(self.selected) <= budget <= len(self.a):
            raise ValueError('预算小于现有前缀或超过查询池')
        while len(self.selected) < budget:
            self.step()
        # 预算边界也刷新，使从已完成前缀恢复与继续运行使用同一数值边界。
        self.refresh()
        return list(self.selected)
