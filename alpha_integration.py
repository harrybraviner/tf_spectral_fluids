#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import functools
import alpha_navier_stokes

def fwd_euler_timestep(x, dx_dt, h):
    """Returns an operation to perform a forward Euler timestep.

    Arguments:
        x: List of tensors whose values are to be updated by the operation.
        dx_dt: A list of tensors (which should be built from x) giving the time derivatives.
        h: Step size

    Returns:
        An operation that performs a single update.
    """

    x_ = [xi + h*dxi_dt for (xi, dxi_dt) in zip(x, dx_dt)]

    return functools.reduce(tf.group, [xi.assign(xi_) for (xi, xi_) in zip(x, x_)])

def run_simulation():
    
    N = 8

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
        v_y[1,   0, 0] = 1j*N
        v_y[N-1, 0, 0] = 1j*N
        v_z = np.zeros(shape=[N, N, N//2 + 1], dtype=np.complex64)

        return [v_x, v_y, v_z]

    v_dft_0 = setup_sinusoid(N)

    v_dft_x = tf.Variable(v_0[0], dtype=tf.complex64)
    v_dft_y = tf.Variable(v_0[1], dtype=tf.complex64)
    v_dft_z = tf.Variable(v_0[2], dtype=tf.complex64)
    v_dft = [v_dft_x, v_dft_y, v_dft_z]
    
    vv_dft = alpha_navier_stokes.velocity_convolution(v_dft)

    k_cmpts = [tf.Variable(k, dtype=tf.float32) for k in alpha_navier_stokes.get_k_cmpts(N, N, N)]
    k_squared = tf.Variable(alpha_navier_stokes.get_k_squared(N, N, N), dtype=tf.float32)
    inv_k_squared = tf.Variable(alpha_navier_stokes.get_inverse_k_squared(N, N, N), dtype=tf.float32)

    # FIXME - Kinda concerned about the consistency of the updates here
    #         i.e. vv_dft depends on v_dft but isn't passed into the lambda as an argument
    #         Should I even be passing v_dft in? Can I just pass the dx_dt tensor??
    #get_v_dft_dt = lambda v_dft : alpha_navier_stokes.eularian_dt(v_dft, vv_dft, 

    #step_op = 
