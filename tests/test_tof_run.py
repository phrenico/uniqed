from unittest import TestCase
from pandas import DataFrame
import numpy as np
from tofpy.runners.tof_run import detect_outlier


class Test(TestCase):
    def test_detect_outlier(self):
        x = DataFrame(np.random.rand(10000))
        df = detect_outlier(x, cutoff_n=1000)
        print(df.head())

