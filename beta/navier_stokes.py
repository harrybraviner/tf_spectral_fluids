import tensorflow as tf
import numpy as np
import unittest

def eularian_dt(v_dft, vv_dft, k_cmpts, k_squared, inverse_k_squared, mask, nu, f_dft):
    """Computes the Eularian derivative of the velocity.
    i.e. \partial_{t'} v = eularian_dt
    
    Arguments:
        v_dft: the DFT of the three velocity components (rank [3, N_x, N_y, N_z//2 + 1] tensor)
        vv_dft: the DFT of v[i]*v[j] (list of 6 tensors, xx, yy, zz, xy, xz, yz).
        k_cmpts: the wavevector components in each directions (a list of three tensors).
        k_squared: the squared magnitude of the wavenumber for each index. Tensor
        inverse_k_squared: the inverse squared magnitude of the wavenumber for each index, with the zero entry masked. Tensor
        mask: The anti-aliasing mask. A [N_x, N_y, N_z//2 + 1] tensor.
        nu: The kinematic viscosity. Should be None for implicit viscosity is used.
        f_dft: The dft of the body force components (list of 3 tensors).

    Returns:
        Time derivates of DFTs of velocity components (a list of 3 tensors).
    """

    complex_type = v_dft[0].dtype.base_dtype

    D_x = eularian_dt_single(v_dft, vv_dft, k_cmpts, k_squared, nu, 0)
    D_y = eularian_dt_single(v_dft, vv_dft, k_cmpts, k_squared, nu, 1)
    D_z = eularian_dt_single(v_dft, vv_dft, k_cmpts, k_squared, nu, 2)

    p_dft = \
        -1j*tf.multiply(
                tf.multiply(tf.cast(k_cmpts[0], dtype=complex_type), D_x) \
              + tf.multiply(tf.cast(k_cmpts[1], dtype=complex_type), D_y) \
              + tf.multiply(tf.cast(k_cmpts[2], dtype=complex_type), D_z),
                tf.cast(inverse_k_squared, dtype=complex_type))
    
    # Negative of pressure gradients (i.e. direction of pressure force)
    neg_p_dx_dft = +1j*tf.multiply(tf.cast(k_cmpts[0], dtype=complex_type), p_dft)
    neg_p_dy_dft = +1j*tf.multiply(tf.cast(k_cmpts[1], dtype=complex_type), p_dft)
    neg_p_dz_dft = +1j*tf.multiply(tf.cast(k_cmpts[2], dtype=complex_type), p_dft)

    if f_dft is not None:
        v_x_dt = D_x + neg_p_dx_dft + f_dft[0]
        v_y_dt = D_y + neg_p_dy_dft + f_dft[1]
        v_z_dt = D_z + neg_p_dz_dft + f_dft[2]
    else:
        v_x_dt = D_x + neg_p_dx_dft
        v_y_dt = D_y + neg_p_dy_dft
        v_z_dt = D_z + neg_p_dz_dft

    # FIXME - need to add anti-aliasing masking to this
    masked_x = tf.multiply(v_x_dt, mask)
    masked_y = tf.multiply(v_y_dt, mask)
    masked_z = tf.multiply(v_z_dt, mask)

    return [masked_x, masked_y, masked_z]

def eularian_dt_single(v_dft, vv_dft, k_cmpts, k_squared, nu, cmpt):
    """The Eularian derivative of the velocity, absent the pressure term,
    for a single component.

    Arguments:
        v_dft: the DFT of the three velocity components (a list of three tensors).
        vv_dft: the DFT of v[i]*v[j] (list of 6 tensors, xx, yy, zz, xy, xz, yz).
        k_cmpts: the wavevector components in each directions (a list of three tensors).
        k_squared: the squared magnitude of the wavenumber for each index. Tensor.
        nu: Kinematic viscosity. Use None to skip the viscous dissipation.
        cmpt : the component of the velocity (0=x, 1=y, 2=z)
    """
    
    def get_vv_cmpt(i, j):
        # Convert to single index
        if i == j:
            return i
        if i > j:
            return get_vv_cmpt(j, i)
        elif i==0 and j==1:
            return 3
        elif i==0 and j==2:
            return 4
        elif i==1 and j==2:
            return 5
        else:
            raise ValueError
            
    complex_type = c_dft[0].dtype.base_dtype

    # Advection
    D_adv = -1j*(tf.multiply(tf.cast(k_cmpts[0], dtype=complex_type), vv_dft[get_vv_cmpt(0, cmpt)]) \
                 + tf.multiply(tf.cast(k_cmpts[1], dtype=complex_type), vv_dft[get_vv_cmpt(1, cmpt)]) \
                 + tf.multiply(tf.cast(k_cmpts[2], dtype=complex_type), vv_dft[get_vv_cmpt(2, cmpt)]))

    # Viscoity
    if nu is not None:
        D_visc = -nu*tf.multiply(v_dft[cmpt],
                                 tf.cast(k_squared, dtype=complex_type),
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
        v_dft: the DFT of the three velocity components (a list of three tensors).
        k_squared: the squared magnitude of the wavenumber for each index. Tensor.
        nu: Kinematic viscosity. Use None to skip the viscous dissipation.
        h: The timestep

    Returns:
        The velocity components decayed by the viscosity (a list of 3 tensors).
    """

    complex_type = v_dft[0].dtype.base_dtype

    decay = tf.cast(tf.exp(-nu*h*k_squared), dtype=complex_type)
    v_dft_x_visc = tf.multiply(decay, v_dft[0])
    v_dft_y_visc = tf.multiply(decay, v_dft[1])
    v_dft_z_visc = tf.multiply(decay, v_dft[2])

    return [v_dft_x_visc, v_dft_y_visc, v_dft_z_visc]

def compute_h_cfl(v_pos, k_cmpts, h_max):
    """Compute timestep from the CFL stability condition.

    Arguments:
        v_pos: A tensor for each of the three position components of velocity (list of three tensors)
        k_cmpts: The wave-vector components (list of three tensors)
        h_max: The maximum step-size (Tensor of shape ())
    Return:
        A rank 0 tensor.
    """

    vx_max = tf.reduce_max(tf.abs(v_pos[0]))
    vy_max = tf.reduce_max(tf.abs(v_pos[1]))
    vz_max = tf.reduce_max(tf.abs(v_pos[2]))
    kx_max = tf.reduce_max(tf.abs(k_cmpts[0]))
    ky_max = tf.reduce_max(tf.abs(k_cmpts[1]))
    kz_max = tf.reduce_max(tf.abs(k_cmpts[2]))

    gamma_v = kx_max*vx_max + ky_max*vx_max + kz_max*vz_max

    if h_max is not None:
        raise NotImplementedError

    return 1.5 / gamma_v


def freq_to_position_space(v_dft):
    """Performs the inverse fourier transforms and returns the velocity
    components in position space
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

    v = [tf.spectral.irfft3d(v_dft[i], name = inv_name(i)) for i in range(3)] 

    return v

def position_space_to_vv_dft(v_pos):
    """Performs multiplications in positions space followed by DFTs to obtain
    the u*u convolution in frequency space.
    """

    def index_to_direction(i):
        if i==0:
            return "x"
        elif i==1:
            return "y"
        elif i==2:
            return "z"

    def fwd_name(i, j):
        return "dft_v_" + index_to_direction(i) + "_v_" + index_to_direction(j)

    conv = [tf.spectral.rfft3d(v_pos[i] * v_pos[j], name = fwd_name(i,j)) for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]]

    return conv

def velocity_convolution(v_dft):
    """The convolution (in wavenumber space) of the velocity components.
    Needed to compute the DFT of the advection term.

    Arguments:
        v_dft: the DFT of the three velocity components (an array of three tensors).
    Returns:
        An array of convolutions of v_dft components.
        Ordering is (x,x), (y,y), (z,z), (x,y), (x,z), (y,z).
        Each entry is a tensor over wavenumbers.
    """

    v_pos = freq_to_position_space(v_dft)
    conv = position_space_to_vv_dft(v_pos)

    return conv
