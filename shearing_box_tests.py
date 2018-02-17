import tensorflow as tf
import numpy as np
import unittest
import shearing_box

class ShearingBoxTests(unittest.TestCase):
    """Note: the tests in this class only check that the
    code actually constructs valid tensors.
    There is no checking that the fields that are produced
    are in fact correct!
    """

    def test_eularian_dt_x_cmpt(self):
        N_x = 16; N_y = 8; N_z = 32
        shape = [N_x, N_y, 1 + N_z//2]

        vx_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        vy_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        vz_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        v_dft = [vx_dft, vy_dft, vz_dft]

        D_x = shearing_box.eularian_dt_single(v_dft, 0.1, 0.2, 0)
        D_y = shearing_box.eularian_dt_single(v_dft, 0.1, 0.2, 1)
        D_z = shearing_box.eularian_dt_single(v_dft, 0.1, 0.2, 2)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        sess.run([D_x, D_y, D_z])

        self.assertEqual(D_x.type, tf.complex64)
        self.assertEqual(D_x.shape.as_list(), shape)
        self.assertEqual(D_y.type, tf.complex64)
        self.assertEqual(D_y.shape.as_list(), shape)
        self.assertEqual(D_z.type, tf.complex64)
        self.assertEqual(D_z.shape.as_list(), shape)
