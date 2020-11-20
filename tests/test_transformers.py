from unittest import TestCase
import numpy as np
from uniqed.transformers.transformers import TimeDelayEmbedder, TransformYTrue, invertit, _make_result_df


class TestTimeDelayEmbedder(TestCase):
    def gen_data(self, n=100):
        return np.random.rand(n)

    def test_fit(self):
        x = self.gen_data()
        TimeDelayEmbedder().fit(x)

    def test_transform(self):
        x = self.gen_data()
        TimeDelayEmbedder().fit(x).transform(x)

    def test_fit_transform(self):
        x = self.gen_data()
        TimeDelayEmbedder().fit(x)

    def test__embedding(self):
        n = 100
        x = self.gen_data()
        d = 3
        tau = 1
        N = n - (d-1)*tau
        shape = [N, d]

        X = TimeDelayEmbedder()._embedding(x, d=d, tau=tau)
        is_eq = np.isclose(shape, X.shape)
        self.assertTrue(np.all(is_eq))


class TestTransformYTrue(TestCase):
    def test_fit(self):
        t = np.zeros(100)
        TransformYTrue().fit(t)

    def test_transform(self):
        t = np.zeros(100)
        TransformYTrue().fit(t).transform(t)

    def test_fit_transform(self):
        t = np.zeros(100)
        TransformYTrue().fit_transform(t)

    def test__transform_y_true(self):
        t = np.zeros(100)
        x = TransformYTrue().fit(t)._transform_y_true(t)

    def test__get_faketime_axis(self):
        n = 5
        t= np.arange(n)
        x = np.zeros(n)
        d = 3
        tau = 1
        fake_t = t[1:-1]
        fake_t_calc = TransformYTrue().fit(x)._get_faketime_axis(d, tau)
        is_eq = np.isclose(fake_t, fake_t_calc)
        self.assertTrue(np.all(is_eq))

    def test_invertit(self):
        x = np.arange(1, 111)
        y = invertit(x, True)
        is_eq = np.isclose(x, 1/y)
        self.assertTrue(np.all(is_eq))


    def test_invertit2(self):
        x = np.arange(1, 111)
        y = invertit(x, False)
        is_eq = np.isclose(x, y)
        self.assertTrue(np.all(is_eq))

    def test__make_result_df(self):
        x = np.arange(1, 111)
        inv_it = True
        prefix = ""

        _make_result_df(x, x, x, inv_it, prefix)
