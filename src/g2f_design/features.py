"""历史唯一实体拟合的有限特征与观察/排名分离。"""
import numpy as np
from .preprocessing import HistoricalPCA, genetic_environment_features


class FoldFeatures:
    def fit(self, historical_combinations, genotypes, environments):
        hybrids = sorted({r['hybrid'] for r in historical_combinations})
        envs = sorted({r['env'] for r in historical_combinations})
        self.genetic_pca = HistoricalPCA(32, genotype=True).fit(np.array([genotypes[g] for g in hybrids]), hybrids)
        self.environment_pca = HistoricalPCA(8).fit(np.array([environments[e] for e in envs]), envs)
        self.genetic = {g: x for g, x in zip(sorted(genotypes), self.genetic_pca.transform(np.array([genotypes[g] for g in sorted(genotypes)])))}
        self.environment = {e: x for e, x in zip(sorted(environments), self.environment_pca.transform(np.array([environments[e] for e in sorted(environments)])))}
        history = self.geometry_components(historical_combinations)
        self.geometry_scale = float(np.sqrt(np.mean(np.einsum('ij,ij->i', history, history))))
        self.geometry_degenerate = self.geometry_scale <= 1e-12
        return self

    def components(self, records, interaction=True):
        genetic = np.array([self.genetic[r['hybrid']] for r in records])
        environment = np.array([self.environment[r['env']] for r in records])
        return genetic_environment_features(genetic, environment, interaction)

    def geometry_components(self, records):
        return np.column_stack(self.components(records, interaction=True))

    def geometry(self, records):
        values = self.geometry_components(records)
        return np.zeros_like(values) if self.geometry_degenerate else values / self.geometry_scale


class ObservationSchema:
    def __init__(self, feature_map, environment_ids, block_ids=(), interaction=True):
        self.feature_map, self.interaction = feature_map, interaction
        self.env_ids = sorted(set(environment_ids))
        self.block_ids = sorted(set(block_ids))
        dg = feature_map.genetic_pca.components.shape[0]
        de = feature_map.environment_pca.components.shape[0]
        sizes = [dg, de] + ([dg * de] if interaction else [])
        self.groups, self.block_names = [-1], []
        self.feature_width = 1 + sum(sizes)
        for name, size in zip(['genetic', 'environment', 'interaction'], sizes):
            if size:
                self.groups.extend([len(self.block_names)] * size)
                self.block_names.append(name)
        self.environment_start = len(self.groups)
        self.groups.extend([len(self.block_names)] * len(self.env_ids))
        self.block_names.append('environment_random_effect')
        self.block_start = len(self.groups)
        if self.block_ids:
            self.groups.extend([len(self.block_names)] * len(self.block_ids))
            self.block_names.append('physical_block_random_effect')
        self.env_lookup = {e: i for i, e in enumerate(self.env_ids)}
        self.block_lookup = {e: i for i, e in enumerate(self.block_ids)}
        self.groups = np.array(self.groups)

    def rows(self, records, observed_blocks=None):
        result = np.zeros((len(records), len(self.groups)), dtype=np.float64)
        result[:, 0] = 1
        result[:, 1:self.feature_width] = np.column_stack(self.feature_map.components(records, self.interaction))
        for i, record in enumerate(records):
            result[i, self.environment_start + self.env_lookup[record['env']]] = 1
        if observed_blocks is not None:
            if len(observed_blocks) != len(records) or not self.block_ids:
                raise ValueError('区组观察输入和已冻结结构不一致')
            for i, block in enumerate(observed_blocks):
                result[i, self.block_start + self.block_lookup[block]] = 1
        return result
