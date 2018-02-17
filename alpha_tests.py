import tensorflow as tf
import numpy as np
import unittest
import alpha

class ShearingBoxTests(unittest.TestCase):
    """Note: the tests in this class only check that the
    code actually constructs valid tensors.
    There is no checking that the fields that are produced
    are in fact correct!
    """

    def test_eularian_dt_x_cmpt(self):
        N_x = 16; N_y = 8; N_z = 32
        shape = [N_x, N_y, 1 + N_z//2]

        nu_k_squared = alpha.get_nu_k_squared(shape, 1.0)

        vx_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        vy_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        vz_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        v_dft = [vx_dft, vy_dft, vz_dft]

        D_x = alpha.eularian_dt_single(v_dft, nu_k_squared, 0)
        D_y = alpha.eularian_dt_single(v_dft, nu_k_squared, 1)
        D_z = alpha.eularian_dt_single(v_dft, nu_k_squared, 2)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        sess.run([D_x, D_y, D_z])

        self.assertEqual(D_x.type, tf.complex64)
        self.assertEqual(D_x.shape.as_list(), shape)
        self.assertEqual(D_y.type, tf.complex64)
        self.assertEqual(D_y.shape.as_list(), shape)
        self.assertEqual(D_z.type, tf.complex64)
        self.assertEqual(D_z.shape.as_list(), shape)

    def test_get_nu_k_squared(self):
        N_x = 16; N_y = 8; N_z = 32
        shape = [N_x, N_y, 1 + N_z//2]

        nu = 3.0

        nu_k_squared = alpha.get_nu_k_squared(N_x, N_y, N_z, nu)

        expected_0 = 0.0
        actual_0 = nu_k_squared[0, 0, 0]
        self.assertEqual(expected_0, actual_0)

        expected_1 = nu*(2.0 * np.pi)**2
        actual_1 = nu_k_squared[1, 0, 0]
        self.assertEqual(expected_1, actual_1)

        expected_2 = nu*(-2.0 * np.pi)**2
        actual_2 = nu_k_squared[N_x-1, 0, 0]
        self.assertEqual(expected_2, actual_2)

        expected_3 = nu*((2.0*np.pi*3.0)**2 + (2.0*np.pi*2.0)**2)
        actual_3 = nu_k_squared[3, 2, 0]
        self.assertEqual(expected_3, actual_3)

        expected_4 = nu*(2.0*np.pi*16)**2
        actual_4 = nu_k_squared[0, 0, 16]
        self.assertEqual(expected_4, actual_4)
