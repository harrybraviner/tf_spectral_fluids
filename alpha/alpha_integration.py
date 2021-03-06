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
        An operation.
    """

    x_ = [xi + h*dxi_dt for (xi, dxi_dt) in zip(x, dx_dt)]
    step_op = functools.reduce(tf.group, [xi.assign(xi_) for (xi, xi_) in zip(x, x_)])

    def run(session):
        step_op.run(session = session)
    return run

def rk3_timestep(x, dx_dt, h):
    """Returns an operation to perform an RK3 timestep.

    Arguments:
        x: List of tensors whose values are to be updated by the operation.
        dx_dt: A list of tensors (which should be built from x) giving the time derivatives.
        h: Step size

    Returns:
        An operation.
    """
    gamma = [8.0/15.0, 5.0/12.0, 3.0/4.0]
    xi = [-17.0/60.0, -5.0/12.0]

    h_cast = tf.cast(h, dtype=tf.complex64)

    # Define variables for our auxilliary storage
    x1 = [tf.Variable(x) for x in x]
    #dx_dt_ = [tf.Variable(d) for d in dx_dt]    # Need this because updating x will update its value

    #step_1_d = functools.reduce(tf.group, [d_.assign(d) for (d_, d) in zip(dx_dt_, dx_dt)])
    #step_1_d = [d_.assign(d) for (d_, d) in zip(dx_dt_, dx_dt)]
    step_1_x = [x.assign(x + gamma[0]*h_cast*d) for (x, d) in zip(x, dx_dt)]
    step_1_x1 = [x1.assign(x + xi[0]*h_cast*d) for (x1, x, d) in zip(x1, step_1_x, dx_dt)]

    step_1_op = functools.reduce(tf.group, step_1_x + step_1_x1)

    #step_2_d = [d_.assign(d) for (d_, d) in zip(dx_dt_, dx_dt)] # This isn't updating properly - why not?
    step_2_x = [x.assign(x1 + gamma[1]*h_cast*d) for (x, x1, d) in zip(x, x1, dx_dt)]
    step_2_x1 = [x1.assign(x + xi[1]*h_cast*d) for (x1, x, d) in zip(x1, step_2_x, dx_dt)]

    step_2_op = functools.reduce(tf.group, step_2_x + step_2_x1)


    #step_3_d = [d_.assign(d) for (d_, d) in zip(dx_dt_, dx_dt)]
    step_3_x = [x.assign(x1 + gamma[2]*h_cast*d) for (x, x1, d) in zip(x, x1, dx_dt)]

    step_3_op = functools.reduce(tf.group, step_3_x)

    #return functools.reduce(tf.group, [step_1_op, step_2_op, step_3_op])
    def run_all(session):
        step_1_op.run(session = session)
        step_2_op.run(session = session)
        step_3_op.run(session = session)

    return run_all

#def implicit_viscosity_step(x, k_squared, nu, h):
#    """Returns an operation which decays the velocity by the viscosity.
#
#    Arguments:
#        x: List of tensors that will be updated by the operation.
#        k_squared: The 


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
    
    N = 32
    nu = 1.0
    #h = 1e-3   # h now set by CFL condition
    t_stop = 3.0
    t_log = 0.1
    step_max = 100

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
    
    # Body force of same magnitude as initial velocity
    f_body = setup_sinusoid(N)

    masks = [tf.Variable(m, dtype=tf.float16)
             for m in alpha_navier_stokes.get_antialiasing_masks(alpha_navier_stokes.get_k_cmpts(N, N, N))]

    vv_dft = alpha_navier_stokes.velocity_convolution(v_dft, masks)

    k_cmpts = [tf.Variable(k, dtype=tf.float32) for k in alpha_navier_stokes.get_k_cmpts(N, N, N)]
    k_squared = tf.Variable(alpha_navier_stokes.get_k_squared(N, N, N), dtype=tf.float32)
    inv_k_squared = tf.Variable(alpha_navier_stokes.get_inverse_k_squared(N, N, N), dtype=tf.float32)

    v_dft_dt = alpha_navier_stokes.eularian_dt(v_dft, vv_dft, k_cmpts, k_squared, inv_k_squared, None, f_body)

    h = tf.Variable(0.0, dtype=tf.float32)
    update_h_by_cfl = h.assign(alpha_navier_stokes.cfl_timestep([N, N, N], v_dft))

    explicit_step_op = rk3_timestep(v_dft, v_dft_dt, h)

    v_dft_post_decay = alpha_navier_stokes.implicit_viscosity(v_dft, k_squared, nu, h)
    implicit_step_op = multi_assign_op(v_dft, v_dft_post_decay)

    #step_op = multi_assign_op(v_dft, v_dft_)

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
    t_next_log = t

    while(t < t_stop):
        h_this_step = update_h_by_cfl.eval(session = sess)  # Set h from CFL condition
        explicit_step_op(session = sess)
        implicit_step_op.run(session = sess)
        t += h_this_step
        timesteps_taken += 1

        if (t >= t_next_log):
            ke = kinetic_energy.eval(session=sess)
            print("t: {}\tKE: {}\th: {}".format(t, ke, h_this_step))
            while(t_next_log <= t):
                t_next_log += t_log

        if (step_max is not None) and (timesteps_taken > step_max):
            break;

    wall_clock_time_at_end = time.time()
    seconds_elapsed = wall_clock_time_at_end - wall_clock_time_at_start
    print('Runtime was {} s'.format(seconds_elapsed))
    print('Time per timestep: {} s'.format(seconds_elapsed / timesteps_taken))

if __name__ == '__main__':
    run_simulation()
