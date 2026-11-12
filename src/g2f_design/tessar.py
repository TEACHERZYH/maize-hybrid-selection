"""固定全池TESSAR和逐步重算的E3预筛；两种L状态严格分开。"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .geometry import NearestAnchors
from .design import best_index
from .randomness import stream


SCALES = (0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01)


class TESSAR:
    def __init__(self, points, identifiers, anchors, anchor_ids, initial, q, configurations,
                 protocol, year, seed, row_chunk=512, anchor_chunk=512, workers=1):
        if configurations not in (50, 100):
            raise ValueError('只允许N=50或100')
        if len(set(identifiers)) != len(identifiers) or len(set(initial)) != len(initial):
            raise ValueError('候选或必选集重复')
        self.all_points = np.asarray(points, dtype=np.float64)
        self.identifiers = list(identifiers)
        self.pool = np.asarray([i for i in range(len(points)) if i not in set(initial)], dtype=int)
        if not 0 <= q <= len(self.pool):
            raise ValueError('固定q超出统计池')
        self.q, self.configurations = q, configurations
        self.context = (protocol, year, seed)
        self.executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self.selected, self.degeneracy_count = list(initial), 0
        self.local_index = {int(v): i for i, v in enumerate(self.pool)}
        pool_points = self.all_points[self.pool]
        self.original = NearestAnchors(pool_points, row_chunk, anchor_chunk)
        self.perturbed = [NearestAnchors(pool_points, row_chunk, anchor_chunk)
                          for _ in range(configurations)]
        self.seen_anchors = set()
        self._add_anchors(np.asarray(anchors), anchor_ids)
        self._add_anchors(self.all_points[initial], [identifiers[i] for i in initial])
        self.l_values = self.fresh_l(np.arange(len(self.pool))) if len(self.pool) else np.empty(0)

    def _add_anchors(self, anchors, identifiers):
        if len(set(identifiers)) != len(identifiers) or self.seen_anchors.intersection(identifiers):
            raise ValueError('锚点ID重复；不同组合可有相同坐标')
        self.seen_anchors.update(identifiers)
        self.original.add(anchors, identifiers)
        repeats = self.configurations // 10

        def update(n):
            nearest = self.perturbed[n]
            scale_index, replicate = divmod(n, repeats)
            noise = np.asarray([stream(*self.context, 'anchor_perturbation', key, scale_index, replicate)
                                .normal(0.0, SCALES[scale_index], size=self.all_points.shape[1])
                                for key in identifiers]).reshape(anchors.shape)
            nearest.add(anchors + noise, identifiers)
        if self.executor is None:
            for n in range(self.configurations):
                update(n)
        else:
            list(self.executor.map(update, range(self.configurations)))

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def fresh_l(self, indices):
        if not len(indices):
            return np.empty(0)
        original = self.original.keys[indices]
        result = np.ones(len(indices))
        for nearest in self.perturbed:
            changed = nearest.keys[indices] != original
            fraction = float(changed.mean())
            result[changed] = np.minimum(result[changed], fraction)
        if np.any(result < 1.0 / len(indices) - 1e-15):
            raise AssertionError('L低于全池定义的下界')
        return result

    def log_scores(self, indices, q, l_values):
        if not 1 <= q <= len(indices):
            raise ValueError('L分位序号非法')
        lq = float(np.partition(l_values, q - 1)[q - 1])
        if lq <= 0:
            raise AssertionError('L_q非正，禁止epsilon修补')
        keys, inverse = np.unique(self.original.keys[indices], return_inverse=True)
        distance2 = self.original.distance2[indices]
        density = np.bincount(inverse, weights=distance2, minlength=len(keys))[inverse]
        eta = np.maximum(l_values - lq, 0) / lq
        with np.errstate(divide='ignore'):
            return -eta + 0.5 * np.log(distance2) + np.log(density)

    def add(self, global_index):
        if global_index not in self.local_index or global_index in self.selected:
            raise ValueError('重复或不在查询统计池的真实选择')
        self.selected.append(global_index)
        self._add_anchors(self.all_points[[global_index]], [self.identifiers[global_index]])
        self.l_values = np.minimum(self.l_values, self.fresh_l(np.arange(len(self.pool))))

    def select(self):
        legal = np.asarray([i not in self.selected for i in self.pool])
        identifiers = [self.identifiers[i] for i in self.pool]
        for _ in range(self.q):
            scores = self.log_scores(np.arange(len(self.pool)), self.q, self.l_values)
            if np.all(np.isneginf(scores[legal])):
                self.degeneracy_count += 1
            chosen = best_index(scores, identifiers, legal)
            legal[chosen] = False
            self.add(int(self.pool[chosen]))
        return self.selected.copy()

    def suggest(self, remaining):
        """E3每步只在当前R统计，L从1初始化；建议不改变任何锚点。"""
        indices = np.asarray([self.local_index[int(i)] for i in remaining], dtype=int)
        q = min(256, len(indices))
        if q == 0:
            return []
        scores = self.log_scores(indices, q, self.fresh_l(indices))
        identifiers = [self.identifiers[i] for i in remaining]
        legal = np.ones(len(indices), dtype=bool)
        chosen = []
        for _ in range(q):
            idx = best_index(scores, identifiers, legal)
            legal[idx] = False
            chosen.append(int(remaining[idx]))
        return chosen


class RandomPrescreen:
    def __init__(self, protocol, year, seed):
        self.context, self.step = (protocol, year, seed), 0

    def suggest(self, remaining):
        return stream(*self.context, 'random_prescreen', str(self.step)).choice(
            remaining, min(256, len(remaining)), replace=False).tolist()

    def add(self, chosen):
        self.step += 1
