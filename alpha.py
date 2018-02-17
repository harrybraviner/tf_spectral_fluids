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

def get_nu_k_squared(shape, nu):
    """The squared magnitude of the wavenumber for each index, multiplied
    by the kinematic viscosity

    Arguments:
        shape: the shape of the tensor to be produced
        nu: the kinematic viscosity
    """

    raise NotImplementedError
    
