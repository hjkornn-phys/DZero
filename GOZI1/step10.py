import numpy as np
import unittest
from GOZI1.step09 import Variable, square


class Squaretest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)