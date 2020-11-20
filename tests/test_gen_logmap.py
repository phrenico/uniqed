from unittest import TestCase
from uniqed.data.gen_logmap import generate_logmapdata, basic, switch
import numpy as np


class Test(TestCase):
    def test_generate_logmapdata(self):
        x = generate_logmapdata()

    def test_generate_logmapdata2(self):
        x = generate_logmapdata(rseed=367)
        y = generate_logmapdata(rseed=367)
        is_eq = np.isclose(x, y)
        self.assertTrue(np.all(is_eq))

    def test_basic(self):
        basic(0.1)

    def test_switch(self):
        switch(0.1)
