#! /usr/bin/python3

import tensorflow as tf
import numpy as np

def eularian_dt(v_dft, vv_dft, k_cmpts, k_squared, inverse_k_squared, nu):
    """Computes the Eularian derivative of the velocity.
    i.e. \partial_{t'} v = eularian_dt
    
    Arguments:
        v_dft: the DFT of the three velocity components (an array of three tensors)
        vv_dft: the DFT of v[i]*v[j] (a 3x3 list of tensors)
        k_cmpts: the wavevector components in each directions (a list of three tensors).
        k_squared: the squared magnitude of the wavenumber for each index. Tensor
        inverse_k_squared: the inverse squared magnitude of the wavenumber for each index, with the zero entry masked. Tensor
        cmpt : the component of the velocity (0=x, 1=y, 2=z)

    Returns:
        Time derivates of DFTs of velocity components (a list of 3 tensors).
    """

    D_x = eularian_dt_single(v_dft, vv_dft, k_cmpts, k_squared, nu, 0)
    D_y = eularian_dt_single(v_dft, vv_dft, k_cmpts, k_squared, nu, 1)
    D_z = eularian_dt_single(v_dft, vv_dft, k_cmpts, k_squared, nu, 2)

    p_dft = \
        -1j*tf.multiply(
                tf.multiply(tf.cast(k_cmpts[0], dtype=tf.complex64), D_x) \
              + tf.multiply(tf.cast(k_cmpts[1], dtype=tf.complex64), D_y) \
              + tf.multiply(tf.cast(k_cmpts[2], dtype=tf.complex64), D_z),
                tf.cast(inverse_k_squared, dtype=tf.complex64))
    
    # Negative of pressure gradients (i.e. direction of pressure force)
    neg_p_dx_dft = +1j*tf.multiply(tf.cast(k_cmpts[0], dtype=tf.complex64), p_dft)
    neg_p_dy_dft = +1j*tf.multiply(tf.cast(k_cmpts[1], dtype=tf.complex64), p_dft)
    neg_p_dz_dft = +1j*tf.multiply(tf.cast(k_cmpts[2], dtype=tf.complex64), p_dft)

    v_x_dt = D_x + neg_p_dx_dft
    v_y_dt = D_y + neg_p_dy_dft
    v_z_dt = D_z + neg_p_dz_dft

    # FIXME - need to add anti-aliasing masking to this

    return [v_x_dt, v_y_dt, v_z_dt]

def velocity_convolution(v_dft):
    """The convolution (in wavenumber space) of the velocity components.
    Needed to compute the DFT of the advection term.

    Arguments:
        v_dft: the DFT of the three velocity components (an array of three tensors).
    Returns:
        An array where conv[i][j] = (convolution of v_dft_i with v_dft_j).
        Each entry is a tensor over wavenumbers.
    """

    def index_to_direction(i):
        if i==0:
            return "x"
        elif i==1:
            return "y"
        elif i==2:
            return "z"

    def inv_name(i):
        return "inverse_dft_v" + index_to_direction(i)

    def fwd_name(i, j):
        return "dft_v_" + index_to_direction(i) + "_v_" + index_to_direction(j)

    v = [tf.spectral.irfft3d(v_dft[i], name = inv_name(i)) for i in range(3)] 
    conv = [[ tf.spectral.rfft3d(v[i] * v[j], name = fwd_name(i,j)) if i <= j else None
             for j in range(3)]
        for i in range(3) ]

    conv = [[conv[i][j] if i <= j else conv[j][i] for j in range(3)]
        for i in range(3) ]

    return conv

def eularian_dt_single(v_dft, vv_dft, k_cmpts, k_squared, nu, cmpt):
    """The Eularian derivative of the velocity, absent the pressure term,
    for a single component.

    Arguments:
        v_dft: the DFT of the three velocity components (a list of three tensors).
        vv_dft: the DFT of v[i]*v[j] (a 3x3 list of tensors)
        k_cmpts: the wavevector components in each directions (a list of three tensors).
        k_squared: the squared magnitude of the wavenumber for each index. Tensor.
        nu: Kinematic viscosity. Use None to skip the viscous dissipation.
        cmpt : the component of the velocity (0=x, 1=y, 2=z)
    """
    
    # Advection
    D_adv = -1j*(tf.multiply(tf.cast(k_cmpts[0], dtype=tf.complex64), vv_dft[0][cmpt]) \
                 + tf.multiply(tf.cast(k_cmpts[1], dtype=tf.complex64), vv_dft[1][cmpt]) \
                 + tf.multiply(tf.cast(k_cmpts[2], dtype=tf.complex64), vv_dft[2][cmpt]))

    # Viscoity
    if nu is not None:
        D_visc = -nu*tf.multiply(v_dft[cmpt],
                                 tf.cast(k_squared, dtype=tf.complex64),
                                 name = "NS_viscosity")

    if nu is not None:
        D = D_adv + D_visc
    else:
        D = D_adv
    return D

def implicit_viscosity(v_dft, k_squared, nu, h):
    """The velocity field decayed by the viscosity.
    This is more stable than using explicit dissipation.

    Arguments:
        v_dft:
        v_dft: the DFT of the three velocity components (a list of three tensors).
        k_squared: the squared magnitude of the wavenumber for each index. Tensor.
        nu: Kinematic viscosity. Use None to skip the viscous dissipation.
        h: The timestep

    Returns:
        The velocity components decayed by the viscosity (a list of 3 tensors).
    """

    decay = tf.cast(tf.exp(-nu*h*k_squared), dtype=tf.complex64)
    v_dft_x_visc = tf.multiply(decay, v_dft[0])
    v_dft_y_visc = tf.multiply(decay, v_dft[1])
    v_dft_z_visc = tf.multiply(decay, v_dft[2])

    return [v_dft_x_visc, v_dft_y_visc, v_dft_z_visc]

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