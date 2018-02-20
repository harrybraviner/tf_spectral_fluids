import tensorflow as tf
import numpy as np
import unittest
import alpha_navier_stokes, alpha_integration

class ShearingBoxTests(unittest.TestCase):
    """Note: the tests in this class only check that the
    code actually constructs valid tensors.
    There is no checking that the fields that are produced
    are in fact correct!
    """

    def assertAlmostEqual(self, expected, actual, eps):
        diff = actual - expected
        if (abs(diff) > eps):
            message = expected + " != " + actual
            raise AssertionError(message)

    def test_convolution(self):
        N_x = 16; N_y = 8; N_z = 32
        shape = [N_x, N_y, 1 + N_z//2]

        k_cmpts = alpha_navier_stokes.get_k_cmpts(N_x, N_y, N_z)
        masks = [tf.Variable(m, dtype=tf.float16) for m in alpha_navier_stokes.get_antialiasing_masks(k_cmpts)]

        vx_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        vy_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        vz_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        v_dft = [vx_dft, vy_dft, vz_dft]

        conv = alpha_navier_stokes.velocity_convolution(v_dft, masks)

        # Check off-diagonal elements are bound to same tensors
        self.assertEqual(conv[0][1].name, conv[1][0].name)
        self.assertEqual(conv[0][2].name, conv[2][0].name)
        self.assertEqual(conv[1][2].name, conv[2][1].name)

        # Check that all tensors are of the expected type
        for i in range(3):
            for j in range(3):
                self.assertEqual(conv[i][j].dtype, tf.complex64)
                self.assertEqual(conv[i][j].shape.as_list(), shape)

    def test_eularian_dt(self):
        N_x = 16; N_y = 8; N_z = 32
        shape = [N_x, N_y, 1 + N_z//2]

        nu = 0.1

        k_squared = tf.Variable(alpha_navier_stokes.get_k_squared(N_x, N_y, N_z), dtype=tf.float32)
        inv_k_squared = tf.Variable(alpha_navier_stokes.get_inverse_k_squared(N_x, N_y, N_z), dtype=tf.float32)
        k_cmpts = alpha_navier_stokes.get_k_cmpts(N_x, N_y, N_z)
        masks = [tf.Variable(m, dtype=tf.float16) for m in alpha_navier_stokes.get_antialiasing_masks(k_cmpts)]

        vx_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        vy_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        vz_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)

        v_dft = [vx_dft, vy_dft, vz_dft]
        vv_dft = alpha_navier_stokes.velocity_convolution(v_dft, masks)
        k_cmpts = [tf.Variable(k, dtype=tf.float32) for k in alpha_navier_stokes.get_k_cmpts(N_x, N_y, N_z)]

        v_dt = alpha_navier_stokes.eularian_dt(v_dft, vv_dft, k_cmpts, k_squared, inv_k_squared, nu)

        v_x_dt, v_y_dt, v_z_dt = v_dt

        self.assertEqual(v_x_dt.dtype, tf.complex64)
        self.assertEqual(v_x_dt.shape.as_list(), shape)
        self.assertEqual(v_y_dt.dtype, tf.complex64)
        self.assertEqual(v_y_dt.shape.as_list(), shape)
        self.assertEqual(v_z_dt.dtype, tf.complex64)
        self.assertEqual(v_z_dt.shape.as_list(), shape)
        
    def test_eularian_dt_cmpts(self):
        N_x = 16; N_y = 8; N_z = 32
        shape = [N_x, N_y, 1 + N_z//2]

        nu = 0.1

        k_squared = tf.Variable(alpha_navier_stokes.get_k_squared(N_x, N_y, N_z), dtype=tf.float32)
        k_cmpts = alpha_navier_stokes.get_k_cmpts(N_x, N_y, N_z)
        masks = [tf.Variable(m, dtype=tf.float16) for m in alpha_navier_stokes.get_antialiasing_masks(k_cmpts)]

        vx_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        vy_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)
        vz_dft = tf.Variable(tf.zeros(shape=shape, dtype=np.complex64), dtype=tf.complex64)

        v_dft = [vx_dft, vy_dft, vz_dft]
        vv_dft = alpha_navier_stokes.velocity_convolution(v_dft, masks)
        k_cmpts = [tf.Variable(k, dtype=tf.float32) for k in alpha_navier_stokes.get_k_cmpts(N_x, N_y, N_z)]

        D_x = alpha_navier_stokes.eularian_dt_single(v_dft, vv_dft, k_cmpts, k_squared, nu, 0)
        D_y = alpha_navier_stokes.eularian_dt_single(v_dft, vv_dft, k_cmpts, k_squared, nu, 1)
        D_z = alpha_navier_stokes.eularian_dt_single(v_dft, vv_dft, k_cmpts, k_squared, nu, 2)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        sess.run([D_x, D_y, D_z])

        self.assertEqual(D_x.dtype, tf.complex64)
        self.assertEqual(D_x.shape.as_list(), shape)
        self.assertEqual(D_y.dtype, tf.complex64)
        self.assertEqual(D_y.shape.as_list(), shape)
        self.assertEqual(D_z.dtype, tf.complex64)
        self.assertEqual(D_z.shape.as_list(), shape)

    def test_get_k_squared(self):
        N_x = 16; N_y = 8; N_z = 32
        shape = [N_x, N_y, 1 + N_z//2]

        k_squared = alpha_navier_stokes.get_k_squared(N_x, N_y, N_z)

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

        inv_k_squared = alpha_navier_stokes.get_inverse_k_squared(N_x, N_y, N_z)
        
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

        k_cmpts = alpha_navier_stokes.get_k_cmpts(N_x, N_y, N_z)

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

        k_cmpts = alpha_navier_stokes.get_k_cmpts(N_x, N_y, N_z)
        masks = alpha_navier_stokes.get_antialiasing_masks(k_cmpts)

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

    def test_forward_euler_timestep(self):
        x = tf.Variable(1.5, dtype=tf.complex64)
        y = tf.Variable(0.0, dtype=tf.complex64)

        dx_dt = [2.0*x, tf.constant(1.0, shape=(), dtype=tf.complex64)]

        h = 0.01

        [x_, y_] = alpha_integration.fwd_euler_timestep([x, y], dx_dt, h)
        step_op = alpha_integration.multi_assign_op([x, y], [x_, y_])

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        step_op.run(session=sess)
        step_op.run(session=sess)

        self.assertAlmostEqual(1.5606+0j, x.eval(session=sess), 1e-6)
        self.assertAlmostEqual(0.02+0j,   y.eval(session=sess), 1e-6)
