"""仅用已知特征的桥接、分层随机与几何覆盖。"""
import numpy as np
from scipy.spatial.distance import cdist
from .design import best_index
from .randomness import stream


def bridge_set(environments, hybrids, identifiers, genetic):
    environments, hybrids = np.asarray(environments), np.asarray(hybrids)
    counts = {g: len(set(environments[hybrids == g])) for g in set(hybrids)}
    selected = []
    for env in sorted(set(environments)):
        indices = list(np.flatnonzero(environments == env))
        if len(indices) < 2:
            raise ValueError('环境查询池不足两个品种')
        first = min(indices, key=lambda i: (-counts[hybrids[i]], identifiers[i]))
        remaining = [i for i in indices if i != first]
        largest = max(counts[hybrids[i]] for i in remaining)
        remaining = [i for i in remaining if counts[hybrids[i]] == largest]
        distances = np.full(len(identifiers), -np.inf)
        distances[remaining] = np.linalg.norm(genetic[remaining] - genetic[first], axis=1)
        legal = np.zeros(len(identifiers), dtype=bool)
        legal[remaining] = True
        selected.extend([first, best_index(distances, identifiers, legal)])
    return selected


def stratified_sequence(environments, identifiers, initial, protocol, year, seed):
    environments = np.asarray(environments)
    selected, initial_set = list(initial), set(initial)
    if len(initial_set) != len(initial):
        raise ValueError('必选集重复')
    queues, capacities, allocated = {}, {}, {}
    for env in sorted(set(environments)):
        indices = sorted((i for i in np.flatnonzero(environments == env) if i not in initial_set),
                         key=lambda i: identifiers[i])
        queues[env] = stream(protocol, year, seed, 'stratified_random', str(env)).permutation(indices).tolist()
        capacities[env], allocated[env] = len(indices), 0
    total = sum(capacities.values())
    for t in range(total):
        legal = [e for e in queues if allocated[e] < capacities[e]]
        # 整数分子保持与逐步欠额公式等价，避免浮点误差破坏同分。
        env = min(legal, key=lambda e: (-((t + 1) * capacities[e] - allocated[e] * total), e))
        selected.append(queues[env][allocated[env]])
        allocated[env] += 1
    return selected


class NearestAnchors:
    """分块欧氏距离；同距按锚点ID，缓存仅保存最近距离和ID。"""
    def __init__(self, points, row_chunk=512, anchor_chunk=512):
        self.points = np.asarray(points, dtype=np.float64)
        self.row_chunk, self.anchor_chunk = row_chunk, anchor_chunk
        self.distance2 = np.full(len(points), np.inf)
        self.keys = np.full(len(points), '', dtype=object)

    def add(self, anchors, identifiers):
        anchors = np.asarray(anchors, dtype=np.float64)
        order = sorted(range(len(anchors)), key=lambda i: identifiers[i])
        for start in range(0, len(order), self.anchor_chunk):
            idx = order[start:start + self.anchor_chunk]
            keys = np.asarray([identifiers[i] for i in idx], dtype=object)
            for offset in range(0, len(self.points), self.row_chunk):
                stop = min(len(self.points), offset + self.row_chunk)
                d = cdist(self.points[offset:stop], anchors[idx], metric='sqeuclidean')
                nearest = d.argmin(axis=1)
                values, chosen = d[np.arange(stop - offset), nearest], keys[nearest]
                old, old_keys = self.distance2[offset:stop], self.keys[offset:stop]
                update = (values < old) | ((values == old) & (chosen < old_keys))
                old[update], old_keys[update] = values[update], chosen[update]


def farthest_sequence(points, identifiers, anchors, anchor_ids, initial, budget):
    if len(set(identifiers)) != len(identifiers) or len(set(initial)) != len(initial):
        raise ValueError('候选或必选集重复')
    if not len(initial) <= budget <= len(points):
        raise ValueError('预算非法')
    nearest = NearestAnchors(points)
    nearest.add(anchors, anchor_ids)
    nearest.add(np.asarray(points)[initial], [identifiers[i] for i in initial])
    legal = np.ones(len(points), dtype=bool)
    legal[initial] = False
    selected = list(initial)
    for _ in range(budget - len(initial)):
        chosen = best_index(np.sqrt(nearest.distance2), identifiers, legal)
        legal[chosen] = False
        selected.append(chosen)
        nearest.add(np.asarray(points)[[chosen]], [identifiers[chosen]])
    return selected
