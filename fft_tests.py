import numpy as np
import tensorflow as tf
import utils
import unittest

class FFTTests(unittest.TestCase):
    """The purpose of these test cases is to ascertain Tensorflow's
    conventions for the discrete fourier transform, and to provide a
    fast way to check that those conventions remain stable between
    versions.

    We are interested in the Fourier transform of real-valued signals,
    in 3d.
    """

    def assert_ndarray_equal(expected, actual, epsilon = 1e-8):
        if (expected.shape != actual.shape):
            raise AssertionError("Shapes of expected and actual do not match.\nExpected: {}\nActual: {}"
                                    .format(expected.shape, actual.shape))
        
        diff = actual - expected
        abs_diff = np.abs(diff)
        if np.max(abs_diff) <= epsilon:
            return
        else:
            num_differing = np.sum(abs_diff > epsilon)
            raise AssertionError(("Expected and actual are not within epsilon of each other.\nDiffer at {} out of {} entries.\n"+\
                                  "Max difference is {}.").
                                    format(num_differing, np.product(expected.shape), np.max(abs_diff)))

    def test_origin_spike_transform(self):
        """Let I be the imaginary unit.
        Let f be a tensor of shape [N_x, N_y, N_z].
        Let f_dft be its discrete Fourier transform.

        This test checks that
        f_dft[i, j, k] = sum_{x, y, z} f[x, y, z] exp(-I*(x*i*h_x + y*j*h_y * z*k*h_z))
        using a 1.0 at (0, 0, 0) and 0.0 elsewhere.
        Here h_x = 2*pi/N_x, etc. for y, z.
        f[x, y, z] is a component of the position-space tensor, not f evaluated at
        (x, y, z) - i.e. they are indices, not coordinates.
        """

        f = tf.placeholder(shape=[16, 16, 16], dtype=tf.float32)
        f_dft = tf.spectral.rfft3d(f)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        f_input = np.zeros(shape = [16, 16, 16], dtype=np.float32)
        f_input[0, 0, 0] = 1.0
        f_actual = sess.run(f_dft, feed_dict = {f : f_input})

        f_expected = np.zeros([16, 16, 9], dtype=np.complex64)
        for (i, j, k) in utils.generate_indices(f_expected.shape):
            f_expected[i, j, k] = np.exp(np.complex(0.0, 0.0))
        FFTTests.assert_ndarray_equal(f_expected, f_actual)

    def test_non_origin_spike_transform(self):
        """As with origin_spike_transform, but there's 7.0 at
        (1.0/(2*pi*N_x), 2.0/(2*pi*N_y), 3.0/(2*pi*N_y)).

        This test is sensitive to the sign convention of the transform.
        """

        f = tf.placeholder(shape=[16, 16, 16], dtype=tf.float32)
        f_dft = tf.spectral.rfft3d(f)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        f_input = np.zeros(shape = [16, 16, 16], dtype=np.float32)
        f_input[1, 2, 3] = 7.0
        f_actual = sess.run(f_dft, feed_dict = {f : f_input})

        f_expected = np.zeros([16, 16, 9], dtype=np.complex64)
        h = (2.0 * np.pi / 16.0)
        for (i, j, k) in utils.generate_indices(f_expected.shape):
            f_expected[i, j, k] = 7.0 * np.exp(np.complex(0.0, -(i*1.0*h + j*2.0*h + k*3.0*h)))
        FFTTests.assert_ndarray_equal(f_expected, f_actual, epsilon=2e-6)

    def test_inverse_spike_transform(self):
        """Test that the inverse function computes f[x, y, z]
        as
        f[x, y, z] = sum_{i, j, k} f_dft[i, j, k] exp(+I*(i*x*h_x + j*y*h_y + k*z*h_z)) / (N_x*N_y*N_z)
        where h_x = 2*pi / N_x, etc. for y and z.

        N.B.: Unlike FFTW, the inverse transform divides by N_x*N_y*N_z.
              We do not need to do this ourselves.
        """

        f_dft = tf.placeholder(shape=[16, 16, 9], dtype = tf.complex64)
        f = tf.spectral.irfft3d(f_dft)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        f_input = np.zeros(shape = [16, 16, 9], dtype=np.complex64)
        f_input[1,2,3] = 2.0
        f_actual = sess.run(f, feed_dict = {f_dft : f_input})

        f_expected = np.zeros([16, 16, 16], dtype=np.float32)
        h = (2.0 * np.pi / 16.0)
        N_total = 16*16*16
        for (x, y, z) in utils.generate_indices(f_expected.shape):
            if (x, y, z) == (0, 0, 0):
                f_expected[x, y, z] = 2.0 / N_total # cos(0) = 1.0
            f_expected[x, y, z] = 2.0 * 2.0 * np.cos(1.0*x*h + 2.0*y*h + 3.0*z*h) / N_total
        FFTTests.assert_ndarray_equal(f_expected, f_actual)
