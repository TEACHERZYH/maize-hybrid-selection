"""用独立全距离矩阵参考核对TESSAR状态，不用其缓存代码构造参考。"""
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist
from g2f_design.randomness import stream
from g2f_design.tessar import TESSAR, SCALES


def direct_reference(points, anchors, anchor_ids, configurations, indices):
    order = sorted(range(len(anchor_ids)), key=lambda i: anchor_ids[i])
    anchors = np.asarray(anchors)[order]
    ids = [anchor_ids[i] for i in order]
    data = np.asarray(points)[indices]
    d2 = cdist(data, anchors, 'sqeuclidean')
    k0 = d2.argmin(axis=1)
    minimum = d2[np.arange(len(data)), k0]
    l_values = np.ones(len(data))
    for n in range(configurations):
        scale, replicate = divmod(n, configurations // 10)
        noise = np.array([stream('P1', 2018, 11, 'anchor_perturbation', key, scale, replicate)
                          .normal(0, SCALES[scale], size=data.shape[1]) for key in ids])
        kn = cdist(data, anchors + noise, 'sqeuclidean').argmin(axis=1)
        changed = kn != k0
        l_values[changed] = np.minimum(l_values[changed], changed.mean())
    density = np.array([minimum[k0 == k].sum() for k in k0])
    return l_values, minimum, density


class TessarTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0.0], [0.003], [0.004], [0.006], [0.007], [0.01], [0.011], [0.02]])
        self.ids = [f'q{i:02d}' for i in range(8)]
        self.anchors, self.anchor_ids = np.array([[-0.001], [0.012]]), ['h0', 'h1']

    def make(self, q=3, configurations=50, **kwargs):
        return TESSAR(self.points, self.ids, self.anchors, self.anchor_ids, [0], q,
                      configurations, 'P1', 2018, 11, **kwargs)

    def test_initial_scores_against_dense_reference_and_persistent_l(self):
        model = self.make()
        anchors = np.vstack((self.anchors, self.points[[0]]))
        ids = self.anchor_ids + [self.ids[0]]
        old = np.ones(7)
        for chosen in (None, 3, 6):
            if chosen is not None:
                model.add(chosen)
                anchors = np.vstack((anchors, self.points[[chosen]]))
                ids.append(self.ids[chosen])
            current, distance2, density = direct_reference(self.points, anchors, ids, 50, model.pool)
            old = np.minimum(old, current)
            assert_allclose(model.l_values, old, atol=0, rtol=0)
            lq = np.sort(old)[model.q - 1]
            with np.errstate(divide='ignore'):
                expected = -np.maximum(old - lq, 0) / lq + 0.5 * np.log(distance2) + np.log(density)
            assert_allclose(model.log_scores(np.arange(7), model.q, model.l_values), expected, atol=1e-12)
            self.assertEqual(model.q, 3)
            self.assertEqual(len(model.pool), 7)

    def test_each_budget_fresh_and_same_seed_chunk_independent(self):
        for n in (50, 100):
            for q in (0, 2, 4):
                a, b = self.make(q, n, row_chunk=1, anchor_chunk=1), self.make(q, n, row_chunk=64, anchor_chunk=64)
                chosen = a.select()
                self.assertEqual(chosen, b.select())
                self.assertEqual(len(chosen), q + 1)
                self.assertEqual(len(set(chosen)), q + 1)

    def test_prescreen_fresh_l_and_no_phantom_anchor(self):
        points = np.linspace(-0.005, 0.03, 270).reshape(-1, 1)
        ids = [f'q{i:03d}' for i in range(270)]
        model = TESSAR(points, ids, self.anchors, self.anchor_ids, [0], 1, 50, 'P1', 2018, 11)
        model.add(100)
        remaining = np.array([i for i in range(270) if i not in (0, 100)])
        anchors = np.vstack((self.anchors, points[[0, 100]]))
        keys = self.anchor_ids + [ids[0], ids[100]]
        l_values, d2, density = direct_reference(points, anchors, keys, 50, remaining)
        local = np.array([model.local_index[int(i)] for i in remaining])
        assert_array_equal(model.fresh_l(local), l_values)
        lq = np.sort(l_values)[255]
        expected_scores = -np.maximum(l_values - lq, 0) / lq + 0.5 * np.log(d2) + np.log(density)
        assert_allclose(model.log_scores(local, 256, model.fresh_l(local)), expected_scores, atol=1e-12)
        before = model.original.keys.copy()
        suggested = model.suggest(remaining)
        self.assertEqual(len(suggested), 256)
        self.assertEqual(len(set(suggested)), 256)
        assert_array_equal(model.original.keys, before)
        self.assertEqual(model.selected, [0, 100])
        self.assertTrue(set(suggested).issubset(remaining))

    def test_zero_geometry_id_fallback_and_no_duplicate_selection(self):
        model = TESSAR(np.zeros((5, 2)), list('abcde'), np.zeros((2, 2)), ['h1', 'h2'],
                       [0], 3, 50, 'P1', 2018, 11)
        self.assertEqual(model.select(), [0, 1, 2, 3])
        self.assertEqual(model.degeneracy_count, 3)
        with self.assertRaises(ValueError):
            model.add(1)

    def test_parallel_configurations_are_bitwise_equal(self):
        with self.make(workers=2) as parallel:
            serial = self.make()
            assert_array_equal(parallel.original.keys, serial.original.keys)
            assert_array_equal(parallel.l_values, serial.l_values)
            self.assertEqual(parallel.select(), serial.select())
