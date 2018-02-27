import tensorflow as tf
import numpy as np
import unittest

def get_k_squared(N_x, N_y, N_z):
    """The squared magnitude of the wavenumber for each index.

    Arguments:
        N_x, N_y, N_z : The number of collocation points in each direction
    Returns:
        An ndarray of shape [N_x, N_y, N_z//2 + 1]
    """

    # This is because of the compressed representation of the DFT
    # of a real-valued field in position space
    def index_mod(i, n):
        if i < n//2:
            return i
        else:
            return i - n

    k_x_2 = np.array([(2.0*np.pi*index_mod(i, N_x))**2 for i in range(N_x)])
    k_y_2 = np.array([(2.0*np.pi*index_mod(j, N_y))**2 for j in range(N_y)])
    k_z_2 = np.array([(2.0*np.pi*k)**2 for k in range(N_z//2 + 1)])

    return \
        k_x_2.reshape([N_x, 1, 1]).repeat(repeats=N_y, axis=1).repeat(repeats=(N_z//2 + 1), axis=2) + \
        k_y_2.reshape([1, N_y, 1]).repeat(repeats=N_x, axis=0).repeat(repeats=(N_z//2 + 1), axis=2) + \
        k_z_2.reshape([1, 1, N_z//2 + 1]).repeat(repeats=N_x, axis=0).repeat(repeats=(N_y), axis=1)

def get_inverse_k_squared(N_x, N_y, N_z):
    """The inverse squared magnitude of the wavenumber for each index.
    The [0, 0, 0] element is 0.0 for masking reasons for the
    pressure gradient calculation.

    Arguments:
        N_x, N_y, N_z : The number of collocation points in each direction
    Returns:
        An ndarray of shape [N_x, N_y, N_z//2 + 1]
    """

    k_sq = get_k_squared(N_x, N_y, N_z)
    k_sq[0, 0, 0] = 1.0 # This is suppress a divide-by-zero warning
    inv_k_sq = 1.0 / k_sq
    inv_k_sq[0, 0, 0] = 0.0

    return inv_k_sq

        
def get_k_cmpts(N_x, N_y, N_z):
    """The x, y, and z components of the wavevectors.

    Arguments:
        N_x, N_y, N_z : The number of collocation points in each direction

    Returns:
        A list of three ndarrays, of shapes [N_x, 1, 1], [1, N_y, 1], and [1, 1, N_z//2 + 1] respectively.
    """

    # This is because of the compressed representation of the DFT
    # of a real-valued field in position space
    def index_mod(i, n):
        if i < n//2:
            return i
        else:
            return i - n

    k_x = np.array([2.0*np.pi*index_mod(i, N_x) for i in range(N_x)]).reshape([N_x, 1, 1])
    k_y = np.array([2.0*np.pi*index_mod(j, N_y) for j in range(N_y)]).reshape([1, N_y, 1])
    k_z = np.array([2.0*np.pi*k for k in range(N_z//2 + 1)]).reshape([1, 1, N_z//2 + 1])

    return [k_x, k_y, k_z]

def get_antialiasing_masks(N_x, N_y, N_z):
    """Masks for anti-aliasing.

    Arguments:
        N_x, N_y, N_z : The number of collocation points in each direction

    Returns:
        A list of three ndarrays, of shapes [N_x, 1, 1], [1, N_y, 1], and [1, 1, N_z//2 + 1] respectively.
    """

    k_cmpts = get_k_cmpts(N_x, N_y, N_z) # Not exactly efficient, but only happens at setup so I don't care

    k_x_max = np.max(abs(k_cmpts[0]))
    k_y_max = np.max(abs(k_cmpts[1]))
    k_z_max = np.max(abs(k_cmpts[2]))

    x_mask = np.array([0.0 if abs(k) > (2.0/3.0)*k_x_max else 1.0 for k in k_cmpts[0].flatten()]).reshape(k_cmpts[0].shape)
    y_mask = np.array([0.0 if abs(k) > (2.0/3.0)*k_y_max else 1.0 for k in k_cmpts[1].flatten()]).reshape(k_cmpts[1].shape)
    z_mask = np.array([0.0 if abs(k) > (2.0/3.0)*k_z_max else 1.0 for k in k_cmpts[2].flatten()]).reshape(k_cmpts[2].shape)

    return [x_mask, y_mask, z_mask]

class WaveVectorTests(unittest.TestCase):

    def test_get_k_squared(self):
        N_x = 16; N_y = 8; N_z = 32
        shape = [N_x, N_y, 1 + N_z//2]

        k_squared = get_k_squared(N_x, N_y, N_z)

        expected_0 = 0.0
        actual_0 = k_squared[0, 0, 0]
        self.assertEqual(expected_0, actual_0)

        expected_1 = (2.0 * np.pi)**2
        actual_1 = k_squared[1, 0, 0]
        self.assertEqual(expected_1, actual_1)

        expected_2 = (-2.0 * np.pi)**2
        actual_2 = k_squared[N_x-1, 0, 0]
        self.assertEqual(expected_2, actual_2)

        expected_3 = (2.0*np.pi*3.0)**2 + (2.0*np.pi*2.0)**2
        actual_3 = k_squared[3, 2, 0]
        self.assertEqual(expected_3, actual_3)

        expected_4 = (2.0*np.pi*16)**2
        actual_4 = k_squared[0, 0, 16]
        self.assertEqual(expected_4, actual_4)

    def test_get_inv_k_squared(self):
        N_x = 16; N_y = 8; N_z = 32
        shape = [N_x, N_y, 1 + N_z//2]

        inv_k_squared = get_inverse_k_squared(N_x, N_y, N_z)
        
        # This should be zero for masking
        expected_0 = 0.0
        actual_0 = inv_k_squared[0, 0, 0]

        expected_1 = 1.0/(2.0 * np.pi)**2
        actual_1 = inv_k_squared[1, 0, 0]
        self.assertEqual(expected_1, actual_1)

        expected_2 = 1.0/(-2.0 * np.pi)**2
        actual_2 = inv_k_squared[N_x-1, 0, 0]
        self.assertEqual(expected_2, actual_2)

        expected_3 = 1.0/((2.0*np.pi*3.0)**2 + (2.0*np.pi*2.0)**2)
        actual_3 = inv_k_squared[3, 2, 0]
        self.assertEqual(expected_3, actual_3)

        expected_4 = 1.0/(2.0*np.pi*16)**2
        actual_4 = inv_k_squared[0, 0, 16]
        self.assertEqual(expected_4, actual_4)
    
    def test_get_k_cmpts(self):
        N_x = 16; N_y = 8; N_z = 32
        shape = [N_x, N_y, 1 + N_z//2]

        k_cmpts = get_k_cmpts(N_x, N_y, N_z)

        self.assertEqual((N_x, 1, 1), k_cmpts[0].shape)
        self.assertEqual((1, N_y, 1), k_cmpts[1].shape)
        self.assertEqual((1, 1, N_z//2 + 1), k_cmpts[2].shape)

        expected_0 = 0.0
        actual_0 = k_cmpts[0][0, 0, 0]
        self.assertEqual(expected_0, actual_0)

        expected_1 = 0.0
        actual_1 = k_cmpts[2][0, 0, 0]
        self.assertEqual(expected_1, actual_1)

        expected_2 = 2.0*np.pi*3.0
        actual_2 = k_cmpts[1][0, 3, 0]
        self.assertEqual(expected_2, actual_2)

        expected_3 = -2.0*np.pi*4.0
        actual_3 = k_cmpts[1][0, 4, 0]
        self.assertEqual(expected_3, actual_3)

        expected_4 = -2.0*np.pi*3.0
        actual_4 = k_cmpts[1][0, 5, 0]
        self.assertEqual(expected_4, actual_4)

        expected_5 = 2.0*np.pi*16.0
        actual_5 = k_cmpts[2][0, 0, 16]
        self.assertEqual(expected_5, actual_5)

    def test_get_antialiasing_masks(self):
        N_x = 16; N_y = 8; N_z = 32
        shape = [N_x, N_y, 1 + N_z//2]

        k_cmpts = get_k_cmpts(N_x, N_y, N_z)
        masks = get_antialiasing_masks(k_cmpts)

        self.assertEqual((N_x, 1, 1), masks[0].shape)
        self.assertEqual((1, N_y, 1), masks[1].shape)
        self.assertEqual((1, 1, N_z//2 + 1), masks[2].shape)

        expected_0 = 1.0
        actual_0 = masks[0][0, 0, 0]
        self.assertEqual(expected_0, actual_0)

        expected_1 = 1.0
        actual_1 = masks[0][5, 0, 0]
        self.assertEqual(expected_1, actual_1)

        expected_2 = 0.0
        actual_2 = masks[0][6, 0, 0]
        self.assertEqual(expected_2, actual_2)

        expected_3 = 0.0
        actual_3 = masks[0][8, 0, 0]
        self.assertEqual(expected_3, actual_3)

        expected_4 = 1.0
        actual_4 = masks[0][11, 0, 0]
        self.assertEqual(expected_4, actual_4)

        expected_5 = 0.0
        actual_5 = masks[0][10, 0, 0]
        self.assertEqual(expected_5, actual_5)

        expected_6 = 1.0
        actual_6 = masks[2][0, 0, 10]
        self.assertEqual(expected_6, actual_6)

        expected_7 = 0.0
        actual_7 = masks[2][0, 0, 11]
        self.assertEqual(expected_7, actual_7)
