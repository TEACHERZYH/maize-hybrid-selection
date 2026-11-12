"""精确缓存须保留稠密参考的选择、区组处理和恢复语义。"""
import sys
import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from g2f_design.gaussian import Posterior
from g2f_design.design import greedy_select, marginal_scores
from g2f_design.fast_design import ExactGreedy


class ExactCacheTests(unittest.TestCase):
    def test_dense_reference_prefixes_and_resume_with_block_nuisance(self):
        rng = np.random.default_rng(120)
        a = rng.normal(size=(360, 16))
        a[11] = a[10]  # 刻意覆盖同分ID情形。
        latent = a.copy()
        latent[:, -4:] = 0  # 潜在f不含观测区组干扰。
        z = rng.normal(size=(16, 16))
        cov = z @ z.T + np.eye(16)
        z = rng.normal(size=(16, 9))
        h = z @ z.T
        prior = Posterior(rng.normal(size=16), cov, 0.02)
        ids = [f'x{360-i:04d}' for i in range(360)]
        initial = [0, 3, 8]
        with threadpool_limits(limits=1):
            for kind in ('M', 'B3', 'B4', 'B5', 'M-static'):
                with self.subTest(kind=kind):
                    reference, expected = greedy_select(prior, a, ids, 300, initial, h=h, kind=kind, latent=latent)
                    engine = ExactGreedy(prior, a, ids, initial, h=h, kind=kind, latent=latent)
                    for budget in (30, 100, 200, 300):
                        self.assertEqual(engine.take(budget), reference[:budget])
                        if budget == 100:
                            resumed = ExactGreedy(prior, a, ids, initial, h=h, kind=kind, latent=latent,
                                                  resume=engine.selected)
                    self.assertEqual(resumed.take(300), reference)
                    assert_allclose(engine.state.covariance, expected.covariance, rtol=1e-12, atol=1e-12)
                    assert_allclose(engine.state.mean, prior.mean, rtol=0, atol=0)
                    if kind in ('M', 'B3', 'B4'):
                        dense_scores = marginal_scores(engine.state.covariance, a, prior.noise, h)
                        assert_allclose(engine.scores(), dense_scores, rtol=1e-10, atol=1e-12)

    def test_zero_geometry_and_invalid_resume(self):
        prior = Posterior(np.zeros(3), np.eye(3), 1e-8)
        a = np.zeros((12, 3))
        ids = [str(20-i) for i in range(12)]
        engine = ExactGreedy(prior, a, ids, [2], h=np.eye(3))
        actual = engine.take(10)
        expected, _ = greedy_select(prior, a, ids, 10, [2], h=np.eye(3))
        self.assertEqual(actual, expected)
        with self.assertRaises(ValueError):
            ExactGreedy(prior, a, ids, [2], h=np.eye(3), resume=[3, 2])


if __name__ == '__main__':
    unittest.main()
