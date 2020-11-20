from unittest import TestCase
import numpy as np
from uniqed.models.tof import TOF
import matplotlib.pyplot as plt


class TestTOF(TestCase):
    def _gen_data(self, n=100, d=5):
        return np.random.random(n*d).reshape([n, d])

    def test_fit(self):
        X = self._gen_data()
        TOF().fit(X)

    def test_predict(self):
        X = self._gen_data()
        TOF().fit(X).predict(X)
        TOF(cutoff_n=70).fit(X).predict(X)


    def test__get_outliers_inds(self):
        X = self._gen_data()
        TOF()._get_outliers_inds(TOF().fit(X).predict(X))

    def test__find_nearest_neighbors(self):
        X = self._gen_data()
        tof = TOF().fit(X)
        tof._find_nearest_neighbors(X)

        tof = TOF().fit(X)
        tof._find_nearest_neighbors(X, k=7)

    def test__compute_cutoff(self):
        X = self._gen_data()
        tof = TOF().fit(X)
        tof._compute_cutoff(cutoff_n=100)
        with self.assertRaises(ValueError):
            tof._compute_cutoff(cutoff_n='goosebump')

    def test__compute_cutoff2(self):
        X = self._gen_data()
        tof = TOF(k=21).fit(X)
        tof._compute_cutoff(cutoff_n=100)


    def test__compute_p_value(self):
        x = np.arange(100)
        p = np.arange(0.01, 1.01, 0.01)
        p_calculated = TOF()._compute_p_value(x)
        is_equal = np.round(p, 2) == np.round(p_calculated, 2)
        self.assertTrue(np.all(is_equal))

    def test__compute_outlier_score(self):
        X = self._gen_data()
        TOF().fit(X)._compute_outlier_score(X)

    def test__compute_tof(self):
        X = self._gen_data()
        nn_ids = np.array([[1, 3], [0, 2], [3,2], [0,1]])
        ids = np.arange(4).reshape([4, 1])
        score = np.mean((nn_ids-ids)**2, axis=1)**(1/2)
        calcscore = TOF().fit(X)._compute_tof(nn_ids, ids)
        is_eq = np.all(score==calcscore)
        self.assertTrue(is_eq)

    def test__compute_perc_cutoff(self):
        X = self._gen_data()

        z = np.arange(1, 101).astype(int)
        cutoff = 10
        perc = 100-cutoff+1
        tof = TOF().fit(X)

        tof.outlier_score_ = z
        perc_cutoff = tof._compute_perc_cutoff(cutoff).astype(int)
        self.assertTrue(perc_cutoff==perc)

