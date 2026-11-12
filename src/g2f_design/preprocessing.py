"""只从历史唯一实体拟合填补、标准化与确定性PCA。"""
import numpy as np


class HistoricalPCA:
    def __init__(self, maximum_components, genotype=False):
        self.maximum_components = maximum_components
        self.genotype = genotype

    def fit(self, values, identifiers):
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('历史PCA必须按唯一实体拟合，不能按重复试验行加权')
        x = np.asarray(values, dtype=np.float64)
        x = np.where(np.isfinite(x), x, np.nan)
        if len(x) != len(identifiers):
            raise ValueError('实体ID和数据行数不符')
        self.source_width = x.shape[1]
        self.available_columns = np.any(np.isfinite(x), axis=0)
        if self.genotype:
            self.available_columns &= np.mean(np.isnan(x), axis=0) <= 0.2
        kept = x[:, self.available_columns]
        self.median = np.nanmedian(kept, axis=0)
        base = np.where(np.isnan(kept), self.median, kept)
        if not self.genotype:
            base = np.column_stack((base, np.isnan(kept).astype(float)))
        self.center = base.mean(axis=0)
        scale = base.std(axis=0, ddof=0)
        # 精确常数列的浮点标准差可能因均值舍入而非零，必须按取值判定。
        self.variable_columns = np.ptp(base, axis=0) > 0
        if np.any(scale[self.variable_columns] == 0):
            raise FloatingPointError('真实变化列的标准差下溢，不能当作常数删除')
        self.scale = scale[self.variable_columns]
        standardized = (base[:, self.variable_columns] - self.center[self.variable_columns]) / self.scale
        if standardized.shape[1] == 0:
            self.components = np.empty((0, 0))
            return self
        _, singular, vectors = np.linalg.svd(standardized, full_matrices=False)
        rank = int(np.sum(singular > singular[0] * 1e-12)) if singular[0] else 0
        vectors = vectors[:rank].copy()
        start = 0
        while start < rank:
            end = start + 1
            while end < rank and abs(singular[end] - singular[start]) <= singular[0] * 1e-12:
                end += 1
            if end - start > 1:
                subspace = vectors[start:end]
                projector = subspace.T @ subspace
                basis = []
                for index in range(projector.shape[1]):
                    value = projector[:, index].copy()
                    for previous in basis:
                        value -= previous @ value * previous
                    norm = np.linalg.norm(value)
                    if norm > 1e-10:
                        basis.append(value / norm)
                    if len(basis) == end - start:
                        break
                if len(basis) != end - start:
                    raise np.linalg.LinAlgError('重复奇异值空间规范化失败')
                vectors[start:end] = np.asarray(basis)
            for index in range(start, end):
                pivot = np.argmax(np.abs(vectors[index]))
                if vectors[index, pivot] < 0:
                    vectors[index] *= -1
            start = end
        self.components = vectors[:min(rank, self.maximum_components)]
        return self

    def transform(self, values):
        x = np.asarray(values, dtype=np.float64)
        if x.shape[1] != self.source_width:
            raise ValueError('待预测特征语义宽度与历史不符')
        x = np.where(np.isfinite(x), x, np.nan)[:, self.available_columns]
        base = np.where(np.isnan(x), self.median, x)
        if not self.genotype:
            base = np.column_stack((base, np.isnan(x).astype(float)))
        standardized = (base[:, self.variable_columns] - self.center[self.variable_columns]) / self.scale
        return standardized @ self.components.T


def genetic_environment_features(genetic, environment, interaction=True):
    g, e = np.asarray(genetic, dtype=float), np.asarray(environment, dtype=float)
    if g.shape[1] == 0:
        raise ValueError('遗传有效秩为0，核心G×E研究不可执行')
    if interaction and e.shape[1] == 0:
        raise ValueError('环境有效秩为0，交互分支不可执行')
    g = g / np.sqrt(g.shape[1])
    e = e / np.sqrt(e.shape[1]) if e.shape[1] else e
    blocks = [g, e]
    if interaction and e.shape[1]:
        blocks.append(np.einsum('ni,nj->nij', g, e).reshape(len(g), -1))
    return blocks
