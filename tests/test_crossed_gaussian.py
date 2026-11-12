"""独立稠密高斯计算核验共享组合项、梯度及区组停用。"""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from g2f_design.gaussian import SufficientStatistics, likelihood
from g2f_design.crossed_gaussian import EnvironmentStatistics, crossed_likelihood, fit_crossed_variances


class CrossedTests(unittest.TestCase):
    def sample(self, blocks=True):
        rng = np.random.default_rng(116)
        groups = np.array([-1, 0, 0, 1])
        statistics, arrays = [], []
        for n in (5, 4):
            x = np.c_[np.ones(n), rng.normal(size=(n, 3))]
            ci = np.repeat(np.arange(n), [2+(i % 2) for i in range(n)])
            bi = np.arange(len(ci)) % 3 if blocks else None
            y = rng.normal(size=len(ci))
            statistics.append(EnvironmentStatistics.from_arrays(x, y, ci, bi))
            arrays.append((x[ci], np.eye(n)[ci], np.eye(3)[bi] if blocks else np.zeros((len(ci), 0)), y))
        total_n = sum(len(a[0]) for a in arrays)
        ncomb = sum(a[1].shape[1] for a in arrays)
        nblock = sum(a[2].shape[1] for a in arrays)
        full = np.zeros((total_n, len(groups)+ncomb+nblock))
        row, combo, block = 0, len(groups), len(groups)+ncomb
        for x, u, z, y in arrays:
            full[row:row+len(x), :len(groups)] = x
            full[row:row+len(x), combo:combo+u.shape[1]] = u
            full[row:row+len(x), block:block+z.shape[1]] = z
            row += len(x); combo += u.shape[1]; block += z.shape[1]
        allgroups = np.r_[groups, np.full(ncomb, 2), np.full(nblock, 3)]
        parameters = np.log([.14, .38, .27]+([.17] if blocks else [])+[.49])
        return statistics, groups, full, allgroups, np.concatenate([a[3] for a in arrays]), parameters

    def test_dense_objective_gradient_and_core_marginal(self):
        for blocks in (False, True):
            stats, groups, x, allgroups, y, p = self.sample(blocks)
            got = crossed_likelihood(p, stats, groups)
            expected = likelihood(p, SufficientStatistics.from_arrays(x, y), allgroups)
            assert_allclose(got[0], expected[0], rtol=2e-11, atol=1e-10)
            assert_allclose(got[1], expected[1], rtol=1e-8, atol=2e-9)
            assert_allclose(got[2].mean, expected[2].mean[:len(groups)], atol=1e-10)
            assert_allclose(got[2].covariance, expected[2].covariance[:len(groups), :len(groups)], atol=1e-10)
            for j in range(len(p)):
                shift = np.eye(len(p))[j] * 1e-5
                fd = (crossed_likelihood(p+shift, stats, groups)[0]-crossed_likelihood(p-shift, stats, groups)[0])/2e-5
                assert_allclose(got[1][j], fd, atol=3e-7)

    def test_fixed_initializations_and_edge_variances(self):
        stats, groups, x, allgroups, y, p = self.sample(True)
        post, fit = fit_crossed_variances(stats, groups)
        self.assertEqual([r['start'] for r in fit['starts']], ['A', 'B', 'C'])
        self.assertTrue(np.isfinite(post.covariance).all())
        for combo, block in ((1e-8, .1), (.1, 1e-8), (100., 100.)):
            p[2:4] = np.log([combo, block])
            got = crossed_likelihood(p, stats, groups)
            expected = likelihood(p, SufficientStatistics.from_arrays(x, y), allgroups)
            assert_allclose(got[0], expected[0], atol=1e-5)
            assert_allclose(got[1], expected[1], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
