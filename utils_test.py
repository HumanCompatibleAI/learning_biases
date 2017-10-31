import unittest
import numpy as np
from utils import Distribution

class TestDistribution(unittest.TestCase):
    def test_equality(self):
        self.assertEqual(Distribution({'a': 0.5, 'b': 0.5}),
                         Distribution({'a': 1, 'b': 1}))
        self.assertEqual(Distribution({'a': 0.5, 'b': 0.5, 'c': 0}),
                         Distribution({'a': 0.5, 'b': 0.5, 'd': 0}))
        self.assertNotEqual(Distribution({'a': 1, 'b': 1}),
                            Distribution({'a': 1}))

    def test_sample(self):
        dist = Distribution({'a': 1, 'b': 1})
        samples = [dist.sample() for _ in range(200)]
        self.assertTrue(samples.count('a') > 10)
        self.assertTrue(samples.count('b') > 10)

    def test_as_numpy_array(self):
        dist = Distribution({'a': 1, 'b': 2, 'd': 2})
        fn = lambda c: ord(c) - ord('a')
        self.assertEqual(list(dist.as_numpy_array(fn)),
                         [0.2, 0.4, 0, 0.4])
        self.assertEqual(list(dist.as_numpy_array(fn, 5)),
                         [0.2, 0.4, 0, 0.4, 0])

if __name__ == '__main__':
    unittest.main()
