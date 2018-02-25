import tensorflow as tf
import numpy as np
import unittest

def get_rk3_op(x, explicit_dx_dt, aux_input, make_aux_input, h):
    """Contruct an RK3 step to integrate x.

    Arguments:
        x: Mutable tensor (tf.Variable).
        explicit_dx_dt: A (tensor, tensor) -> tensor that we pass x and aux_x into to get dx_dt.
        aux_input: inputs to dx_dt that we have pre-computed for the first step.
        make_aux_input: tensor -> tensor that gets us aux_input from x for subsequent steps.
        h: step size.

    Returns:
        An operation that can be run to step the mutable variable x forward in time.
    """

    gamma = [8.0/15.0, 5.0/12.0, 3.0/4.0]
    xi = [-17.0/60.0, -5.0/12.0]

    # Two storage variables of the same shape as x
    D = tf.Variable(x)
    x1 = tf.Variable(x)

    # Step 1
    if aux_input is not None:
        step1_D = D.assign(explicit_dx_dt(x, aux_input))
    else:
        step1_D = D.assign(explicit_dx_dt(x))
    with tf.control_dependencies([step1_D]):
        step1_x = x.assign(x + gamma[0]*h*D)
        with tf.control_dependencies([step1_x]):
            step1_x1 = x1.assign(x + xi[0]*h*D)
    step1_op = tf.group(step1_D, step1_x, step1_x1)


    # Step 2
    with tf.control_dependencies([step1_x]):
        if aux_input is not None:
            step2_aux_input = make_aux_input(x)
            step2_D = D.assign(explicit_dx_dt(step1_x, step2_aux_input))
        else:
            step2_D = D.assign(explicit_dx_dt(step1_x))
    with tf.control_dependencies([step2_D]):
        step2_x = x.assign(x1 + gamma[1]*h*D)
        with tf.control_dependencies([step2_x]):
            step2_x1 = x1.assign(x + xi[1]*h*D)
    with tf.control_dependencies([step1_op]):
        step2_op = tf.group(step2_D, step2_x, step2_x1)

    # Step 3
    with tf.control_dependencies([step2_x]):
        if aux_input is not None:
            step3_aux_input = make_aux_input(x)
            step3_D = D.assign(explicit_dx_dt(x, step3_aux_input))
        else:
            step3_D = D.assign(explicit_dx_dt(x))
    with tf.control_dependencies([step3_D]):
        step3_x = x.assign(x1 + gamma[2]*h*D)
    with tf.control_dependencies([step2_op]):
        step3_op = tf.group(step3_D, step3_x)

    return tf.group(step3_op)

class IntegratorTests(unittest.TestCase):

    def test_rk3(self):
        h = 0.01

        def dx_dt_imp(x):
            """Imperitive version of time derivative"""
            return np.array([2.0*x[0], 1.0])

        gamma = [8.0/15.0, 5.0/12.0, 3.0/4.0]
        xi = [-17.0/60.0, -5.0/12.0]

        def rk3_imp(x):
            """Imperitive version of rk3"""
            # Step 1
            D = dx_dt_imp(x)
            x = x + gamma[0] * h * D
            x1 = x + xi[0] *h * D
            # Step 2
            D = dx_dt_imp(x)
            x = x1 + gamma[1] * h * D
            x1 = x + xi[1] * h * D
            # Step 3
            D = dx_dt_imp(x)
            x = x1 + gamma[2] * h * D
            return x

        x = tf.Variable([1.5, 0.0], dtype=tf.complex64)

        def dx_dt(x):
            return tf.stack([2.0*x[0], 1.0])

        rk3_op = get_rk3_op(x, dx_dt, None, None, h)

        x_imp = np.array([1.5, 0.0], dtype=np.complex64)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        expected_0 = x_imp
        actual_0 = x.eval(session = sess)

        self.assertEqual(expected_0[0], actual_0[0])
        self.assertEqual(expected_0[1], actual_0[1])

        rk3_op.run(session = sess)

        expected_1 = rk3_imp(x_imp)
        actual_1 = x.eval(session=sess)

        self.assertAlmostEqual(expected_1[0], actual_1[0], 6)
        self.assertAlmostEqual(expected_1[1], actual_1[1], 6)

        rk3_op.run(session = sess)

        expected_2 = rk3_imp(expected_1)
        actual_2 = x.eval(session=sess)

        self.assertAlmostEqual(expected_2[0], actual_2[0], 6)
        self.assertAlmostEqual(expected_2[1], actual_2[1], 6)
        
    def test_rk3_with_aux_info(self):
        """The point of tihs test is to ascertain that auxilliary information
        is correctly computed and used within the RK3 stepping.

        The reason we want this ability is that in the spectral method we will
        compute the position-space velocity field for the purpose of computing
        the timestep before the RK3 step is called. We therefore want to re-use
        this to reduce the number of DFTs we do.
        """

        h = 0.01

        def dx_dt_imp(x):
            """Imperitive version of time derivative"""
            return np.array([2.0*x[0], x[1]*x[1]])

        gamma = [8.0/15.0, 5.0/12.0, 3.0/4.0]
        xi = [-17.0/60.0, -5.0/12.0]

        def rk3_imp(x):
            """Imperitive version of rk3"""
            # Step 1
            D = dx_dt_imp(x)
            x = x + gamma[0] * h * D
            x1 = x + xi[0] *h * D
            # Step 2
            D = dx_dt_imp(x)
            x = x1 + gamma[1] * h * D
            x1 = x + xi[1] * h * D
            # Step 3
            D = dx_dt_imp(x)
            x = x1 + gamma[2] * h * D
            return x

        x_imp = np.array([1.5, -1.0])
        
        x = tf.Variable([1.5, -1.0], dtype=tf.complex64)
        aux_input = tf.Variable((-1.0*-1.0), dtype=tf.complex64)

        def make_aux_input(x):
            return x[1]*x[1]

        def dx_dt(x, aux_input):
            return tf.stack([2.0*x[0], aux_input])

        rk3_op = get_rk3_op(x, dx_dt, aux_input, make_aux_input, h)
        update_aux_input_op = tf.group(aux_input.assign(make_aux_input(x)))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        expected_0 = x_imp
        actual_0 = x.eval(session = sess)

        rk3_op.run(session = sess)
        update_aux_input_op.run(session=sess)
        expected_1 = rk3_imp(x_imp)
        actual_1 = x.eval(session = sess)

        rk3_op.run(session = sess)
        update_aux_input_op.run(session=sess)
        expected_2 = rk3_imp(expected_1)
        actual_2 = x.eval(session = sess)

        rk3_op.run(session = sess)
        update_aux_input_op.run(session=sess)
        expected_3 = rk3_imp(expected_2)
        actual_3 = x.eval(session = sess)

        self.assertAlmostEqual(expected_0[0], actual_0[0], 6)
        self.assertAlmostEqual(expected_0[1], actual_0[1], 6)
        self.assertAlmostEqual(expected_1[0], actual_1[0], 6)
        self.assertAlmostEqual(expected_1[1], actual_1[1], 6)
        self.assertAlmostEqual(expected_2[0], actual_2[0], 6)
        self.assertAlmostEqual(expected_2[1], actual_2[1], 6)
        self.assertAlmostEqual(expected_3[0], actual_3[0], 6)
        self.assertAlmostEqual(expected_3[1], actual_3[1], 6)
