import numpy as np
import utils
import unittest

class UtilsTests(unittest.TestCase):

    def test_generate_indices(self):
        shape_1 = (3,)
        expected_1 = [(0,), (1,), (2,)]
        actual_1 = [x for x in utils.generate_indices(shape_1)]
        self.assertEqual(expected_1, actual_1)

        shape_2 = (2, 3)
        expected_2 = [(0, 0), (0, 1), (0, 2),
                      (1, 0), (1, 1), (1, 2)]
        actual_2 = [x for x in utils.generate_indices(shape_2)]
        self.assertEqual(expected_2, actual_2)

        shape_3 = (4, 1, 2)
        expected_3 = [(0, 0, 0), (0, 0, 1),
                      (1, 0, 0), (1, 0, 1),
                      (2, 0, 0), (2, 0, 1),
                      (3, 0, 0), (3, 0, 1)]
        actual_3 = [x for x in utils.generate_indices(shape_3)]
        self.assertEqual(expected_3, actual_3)
