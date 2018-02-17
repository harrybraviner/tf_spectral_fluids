#! /usr/bin/python3

import tensorflow as tf
import numpy as np

def eularian_dt(v_dft, a, a_dot):
    """Computes the Eularian derivative of the velocity.
    i.e. \partial_{t'} v = eularian_dt
    
    Arguments:
        v_dft: the DFT of the three velocity components (an array of three tensors)
        a : the current shear amplitude
        a_dot : the current time derivative of the shear
    """
    raise NotImplementedError

def eularian_dt_no_pressure(v_dft):
    """The Eularian derivative of the velocity, absent the pressure term.
    This is because we need to know all 3 components of this field before we
    are able to compute the pressure field.
    
    Arguments:
        v_dft: the DFT of the three velocity components (an array of three tensors)
    """
    raise NotImplementedError

def eularian_dt_single(v_dft, nu_k_squared, cmpt):
    """The Eurlarian derivative of the velocity, absent the pressure term,
    for a single component.

    Arguments:
        v_dft: the DFT of the three velocity components (an array of three tensors)
        nu_k_squared: The squared magnitude of the wavenumber for each index,
                      multiplied by the kinematic viscosity.
        cmpt : the component of the velocity (0=x, 1=y, 2=z)
    """


    raise NotImplementedError

def get_nu_k_squared(N_x, N_y, N_z, nu):
    """The squared magnitude of the wavenumber for each index, multiplied
    by the kinematic viscosity

    Arguments:
        N_x, N_y, N_z : The number of collocation points in each direction
        nu: the kinematic viscosity
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
        nu * (k_x_2.reshape([N_x, 1, 1]).repeat(repeats=N_y, axis=1).repeat(repeats=(N_z//2 + 1), axis=2) + \
        k_y_2.reshape([1, N_y, 1]).repeat(repeats=N_x, axis=0).repeat(repeats=(N_z//2 + 1), axis=2) + \
        k_z_2.reshape([1, 1, N_z//2 + 1]).repeat(repeats=N_x, axis=0).repeat(repeats=(N_y), axis=1))
        
