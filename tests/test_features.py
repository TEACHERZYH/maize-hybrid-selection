"""无交互模型、真实区组干扰与历史特征拟合边界。"""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from g2f_design.features import FoldFeatures, ObservationSchema
from g2f_design.gaussian import SufficientStatistics, likelihood
from g2f_design.preprocessing import genetic_environment_features


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.history = [{'env': e, 'hybrid': g} for e in ('e1', 'e2', 'e3') for g in ('g1', 'g2', 'g3')]
        self.genotypes = {'g1': [0, 1, 2], 'g2': [1, 0, 2], 'g3': [2, 2, 0], 'new': [0, 0, 0]}
        self.environments = {'e1': [0, 1], 'e2': [1, 0], 'e3': [2, 2], 'future': [999, -999]}
        self.fmap = FoldFeatures().fit(self.history, self.genotypes, self.environments)

    def test_unseen_entities_do_not_change_historical_fit(self):
        subset = FoldFeatures().fit(self.history, {k: v for k, v in self.genotypes.items() if k != 'new'},
                                     {k: v for k, v in self.environments.items() if k != 'future'})
        assert_allclose(self.fmap.genetic_pca.components, subset.genetic_pca.components, atol=0)
        assert_allclose(self.fmap.environment_pca.components, subset.environment_pca.components, atol=0)
        self.assertEqual(self.fmap.geometry_scale, subset.geometry_scale)

    def test_block_in_observation_but_not_latent_rank(self):
        schema = ObservationSchema(self.fmap, self.environments, ['b1', 'b2'])
        records = [{'env': 'future', 'hybrid': 'g1'}, {'env': 'future', 'hybrid': 'g2'}]
        latent = schema.rows(records)
        observed = schema.rows(records, ['b1', 'b2'])
        assert_array_equal(latent[:, schema.block_start:], 0)
        assert_array_equal(observed[:, schema.block_start:], np.eye(2))
        self.assertEqual((latent[0] - latent[1])[schema.environment_start:].max(), 0)
        self.assertEqual(schema.block_names[-1], 'physical_block_random_effect')

    def test_no_interaction_refits_distinct_sufficient_statistics(self):
        full = ObservationSchema(self.fmap, self.environments)
        reduced = ObservationSchema(self.fmap, self.environments, interaction=False)
        self.assertLess(len(reduced.groups), len(full.groups))
        self.assertNotIn('interaction', reduced.block_names)
        x = reduced.rows(self.history)
        statistics = SufficientStatistics.from_arrays(x, np.arange(9.0))
        nll, _, post = likelihood(np.log([0.1] * len(reduced.block_names) + [0.5]), statistics, reduced.groups)
        self.assertTrue(np.isfinite(nll))
        self.assertEqual(len(post.mean), len(reduced.groups))

    def test_streamed_statistics_equal_single_small_matrix(self):
        schema = ObservationSchema(self.fmap, self.environments, ['b1', 'b2'])
        x = schema.rows(self.history, ['b1'] * 5 + ['b2'] * 4)
        y = np.arange(9.0)
        streamed = SufficientStatistics.from_arrays(x[:4], y[:4])
        streamed.add(x[4:], y[4:])
        dense = SufficientStatistics.from_arrays(x, y)
        assert_allclose(streamed.xx, dense.xx, atol=1e-12)
        assert_allclose(streamed.xy, dense.xy, atol=1e-12)
        self.assertEqual(streamed.n, dense.n)
        self.assertEqual(streamed.yy, dense.yy)

    def test_environment_zero_rank_blocks_interaction(self):
        with self.assertRaises(ValueError):
            genetic_environment_features(np.ones((4, 2)), np.empty((4, 0)), interaction=True)
        self.assertEqual(len(genetic_environment_features(np.ones((4, 2)), np.empty((4, 0)), interaction=False)), 2)
