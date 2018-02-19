#! /usr/bin/python3

import tensorflow as tf
import numpy as np
import functools
import alpha_navier_stokes

def fwd_euler_timestep(x, get_dx_dt, h):
    """Returns an operation to perform a forward Euler timestep.

    Arguments:
        x: List of tensors whose values are to be updated by the operation.
        get_dx_dt: [tensor] -> [tensor] function giving the time derivatives.
        h: Step size

    Returns:
        An operation that performs a single update.
    """

    x_dt = get_dx_dt(x)
    x_ = [xi + h*xi_dt for (xi, xi_dt) in zip(x, x_dt)]

    return functools.reduce(tf.group, [xi.assign(xi_) for (xi, xi_) in zip(x, x_)])

