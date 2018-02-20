#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import functools, time
import alpha_navier_stokes

def fwd_euler_timestep(x, dx_dt, h):
    """Returns an operation to perform a forward Euler timestep.

    Arguments:
        x: List of tensors whose values are to be updated by the operation.
        dx_dt: A list of tensors (which should be built from x) giving the time derivatives.
        h: Step size

    Returns:
        A list of tensors respresenting the forward-stepped values.
    """

    x_ = [xi + h*dxi_dt for (xi, dxi_dt) in zip(x, dx_dt)]
    return x_


def multi_assign_op(x, x_):
    """Assigns the tensors in x_ to the tensors in x

    Arguments:
        x : A list of tensors
        x_ : A list of tensors, each of the same shape as those in x

    Returns:
        An operation performing an update
    """

    return functools.reduce(tf.group, [xi.assign(xi_) for (xi, xi_) in zip(x, x_)])



def run_simulation():
    
    N = 256
    nu = 1.0
    h = 1e-3
    t_stop = 10.0
    t_log = 0.1

    def setup_sinusoid(N):
        """Returns the dft of the velocity field
           u_y = sin(2*pi*x)

        Arguments:
            N: The number of collocation points in each direction

        Returns:
            The DFT of the velocity field as a list of three numpy arrays
        """
        
        v_x = np.zeros(shape=[N, N, N//2 + 1], dtype=np.complex64)
        v_y = np.zeros(shape=[N, N, N//2 + 1], dtype=np.complex64)
        v_y[1,   0, 0] = +0.5j*N*N*N
        v_y[N-1, 0, 0] = -0.5j*N*N*N
        v_z = np.zeros(shape=[N, N, N//2 + 1], dtype=np.complex64)

        return [v_x, v_y, v_z]

    v_dft_0 = setup_sinusoid(N)

    v_dft_x = tf.Variable(v_dft_0[0], dtype=tf.complex64)
    v_dft_y = tf.Variable(v_dft_0[1], dtype=tf.complex64)
    v_dft_z = tf.Variable(v_dft_0[2], dtype=tf.complex64)
    v_dft = [v_dft_x, v_dft_y, v_dft_z]
    
    masks = [tf.Variable(m, dtype=tf.float16)
             for m in alpha_navier_stokes.get_antialiasing_masks(alpha_navier_stokes.get_k_cmpts(N, N, N))]

    vv_dft = alpha_navier_stokes.velocity_convolution(v_dft, masks)

    k_cmpts = [tf.Variable(k, dtype=tf.float32) for k in alpha_navier_stokes.get_k_cmpts(N, N, N)]
    k_squared = tf.Variable(alpha_navier_stokes.get_k_squared(N, N, N), dtype=tf.float32)
    inv_k_squared = tf.Variable(alpha_navier_stokes.get_inverse_k_squared(N, N, N), dtype=tf.float32)

    v_dft_dt = alpha_navier_stokes.eularian_dt(v_dft, vv_dft, k_cmpts, k_squared, inv_k_squared, None)

    v_dft_after_explicit_update = fwd_euler_timestep(v_dft, v_dft_dt, h)
    v_dft_ = alpha_navier_stokes.implicit_viscosity(v_dft_after_explicit_update, k_squared, nu, h)

    step_op = multi_assign_op(v_dft, v_dft_)

    def get_energy(field_dft):
        # Need to double-count the k_z != 0 components due to the half-real representation
        # Would this be more efficient as a broadcast of [[[1.0, 2.0, 2.0, ..., 2.0]]]?
        a = 2.0*tf.reduce_sum(tf.multiply(field_dft, tf.conj(field_dft)))
        b = tf.reduce_sum(tf.multiply(field_dft[:, :, 0], tf.conj(field_dft[:, :, 0])))
        return (1.0/(2.0*(N*N*N)**2))*(a - b)

    kinetic_energy_x = get_energy(v_dft_x)
    kinetic_energy_y = get_energy(v_dft_y)
    kinetic_energy_z = get_energy(v_dft_z)
    kinetic_energy = kinetic_energy_x + kinetic_energy_y + kinetic_energy_z

    #sess = tf.Session()
    sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU' : 1}))
    sess.run(tf.global_variables_initializer())

    wall_clock_time_at_start = time.time()
    timesteps_taken = 0

    t = 0.0
    t_next_log = t + t_log

    while(t < t_stop):
        step_op.run(session = sess)
        t += h
        timesteps_taken += 1

        if (t >= t_next_log):
            ke = kinetic_energy.eval(session=sess)
            print("t: {}\tKE: {}".format(t, ke))
            while(t_next_log <= t):
                t_next_log += t_log

    wall_clock_time_at_end = time.time()
    seconds_elapsed = wall_clock_time_at_end - wall_clock_time_at_start
    print('Runtime was {} s'.format(seconds_elapsed))
    print('Time per timestep: {} s'.format(seconds_elapsed / timesteps_taken))

if __name__ == '__main__':
    run_simulation()
