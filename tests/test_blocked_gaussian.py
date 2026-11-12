"""历史区组消元对稠密参考、差分梯度和查询后验的独立核验。"""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from g2f_design.gaussian import SufficientStatistics, likelihood, fit_variances
from g2f_design.blocked_gaussian import BlockedStatistics, blocked_likelihood, fit_blocked_variances, append_independent_blocks
from g2f_design.design import objective_h
from g2f_design.fast_design import ExactGreedy


class BlockedGaussianTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(73)
        self.x = np.column_stack([np.ones(90), self.rng.normal(size=(90, 6))])
        self.groups = np.array([-1, 0, 0, 1, 1, 2, 2])
        self.blocks = np.arange(90) % 17
        self.u = np.eye(17)[self.blocks]
        self.y = self.x @ self.rng.normal(size=7) + self.u @ self.rng.normal(size=17) + self.rng.normal(size=90) * .3
        self.statistics = BlockedStatistics.empty(7, 17)
        for start in range(0, 90, 13):
            self.statistics.add(self.x[start:start+13], self.y[start:start+13], self.blocks[start:start+13])
        self.dense_statistics = SufficientStatistics.from_arrays(np.c_[self.x, self.u], self.y)
        self.dense_groups = np.r_[self.groups, np.full(17, 3)]

    def test_exact_likelihood_gradient_and_marginal(self):
        for parameters in ([.3, .4, .1, .9, .2], [1e-3, 2., .04, 1e-5, .7], [2., .1, .3, 1., 1e-4]):
            log = np.log(parameters)
            value, gradient, posterior = blocked_likelihood(log, self.statistics, self.groups)
            dense_value, dense_gradient, dense = likelihood(log, self.dense_statistics, self.dense_groups)
            np.testing.assert_allclose(value, dense_value, rtol=1e-9, atol=2e-7)
            np.testing.assert_allclose(gradient, dense_gradient, rtol=2e-7, atol=2e-6)
            np.testing.assert_allclose(posterior.mean, dense.mean[:7], rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(posterior.covariance, dense.covariance[:7, :7], rtol=1e-7, atol=1e-10)
        log = np.log([.3, .4, .1, .9, .2])
        _, gradient, _ = blocked_likelihood(log, self.statistics, self.groups)
        difference = []
        for i in range(len(log)):
            shift = np.zeros(len(log)); shift[i] = 1e-5
            difference.append((blocked_likelihood(log+shift, self.statistics, self.groups)[0]
                               - blocked_likelihood(log-shift, self.statistics, self.groups)[0]) / 2e-5)
        np.testing.assert_allclose(gradient, difference, rtol=2e-6, atol=2e-6)

    def test_query_blocks_selection_and_mean_update(self):
        log = np.log([.3, .4, .1, .9, .2])
        _, _, marginal = blocked_likelihood(log, self.statistics, self.groups)
        reduced = append_independent_blocks(marginal, 3, .9)
        all_x = np.c_[self.x, self.u, np.zeros((90, 3))]
        _, _, dense = likelihood(log, SufficientStatistics.from_arrays(all_x, self.y), np.r_[self.dense_groups, [3]*3])
        query_core = np.column_stack([np.ones(12), self.rng.normal(size=(12, 6))])
        query_blocks = np.eye(3)[np.arange(12) % 3]
        short = np.c_[query_core, query_blocks]
        full = np.c_[query_core, np.zeros((12, 17)), query_blocks]
        xa = np.c_[query_core, np.zeros((12, 3))]
        full_xa = np.c_[query_core, np.zeros((12, 20))]
        ids = ['g%02d' % i for i in range(12)]
        envs = ['e0'] * 6 + ['e1'] * 6
        h, _ = objective_h(reduced, xa, ids, envs, kind='M')
        hf, _ = objective_h(dense, full_xa, ids, envs, kind='M')
        keep = list(range(7)) + list(range(24, 27))
        np.testing.assert_allclose(h, hf[np.ix_(keep, keep)], rtol=1e-8, atol=1e-8)
        chosen = ExactGreedy(reduced, short, ids, [0, 6], h=h, kind='M', latent=xa).take(7)
        dense_chosen = ExactGreedy(dense, full, ids, [0, 6], h=hf, kind='M', latent=full_xa).take(7)
        self.assertEqual(chosen, dense_chosen)
        y = self.rng.normal(size=len(chosen))
        result = reduced.condition_labels(short[chosen], y)
        reference = dense.condition_labels(full[chosen], y)
        np.testing.assert_allclose(result.mean, reference.mean[keep], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(result.covariance, reference.covariance[np.ix_(keep, keep)], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(xa @ result.mean, full_xa @ reference.mean, rtol=1e-8, atol=1e-8)

    def test_frozen_three_start_fit(self):
        posterior, receipt = fit_blocked_variances(self.statistics, self.groups)
        dense, dense_receipt = fit_variances(self.dense_statistics, self.dense_groups)
        self.assertEqual([r['start'] for r in receipt['starts']], ['A', 'B', 'C'])
        self.assertTrue(any(r['success'] for r in receipt['starts']))
        np.testing.assert_allclose(receipt['nll'], dense_receipt['nll'], atol=1e-6, rtol=0)
        np.testing.assert_allclose(posterior.mean, dense.mean[:7], atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
