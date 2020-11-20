from unittest import TestCase
from pandas import DataFrame
import numpy as np
from uniqed.runners.tof_run import detect_outlier


class Test(TestCase):
    def test_detect_outlier(self):
        x = DataFrame(np.random.rand(10000))
        df1 = detect_outlier(x, cutoff_n=1000)
        df2 = detect_outlier(x, cutoff_n=1, in_percent=True)


