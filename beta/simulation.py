#! /usr/bin/python3

import tensorflow as tf
import numpy as np

N = 32
real_type = tf.float32
t_start = 0.0
steps_max = None


if real_type == tf.float32:
    complex_type = tf.complex64
elif real_type = tf.float64:
    complex_type = tf.complex128
else:
    raise ValueError("Unrecognized real_type.")

v_dft = [tf.Variable(v, dtype = complex_type) for v in init_flow.get_sinusoid(N)]
# Components of the velocity convolution
# Ordering convention is [xx, yy, zz, xy, xz, yz]
vv_dft = [tf.zeros(shape = v_dft[0].shape, dtype = complex_type) for _ in range(6)]
# Position space representation of the velocity field
v_pos = tf.Variable(v_dft, dtype = real_type)
# Components of the wavevectors
k_cmpts = [tf.Variable(k, dtype=real_type) for k in wavevector.get_k_cmpts(N, N, N)]

h = tf.Variable(0, dtype=real_type) # Time-step size
t = tf.Variable(t_start, dtype=real_type)
step_count = tf.Variable(0, dtype=tf.int32)

#refresh_vv_dft_op = reduce(tf.group, [vv_dft.assign(x) for (vv_dft, x) in zip(vv_dft, navier_stokes.velocity_convolution())])

## Sequence (can I wrap in a tf.while_loop? At least make it all one op.)
# 1) Compute v_pos
# 2) Compute h_cfl
# 3) Compute vv_dft
# 4) Outputs (you have the position space field and vv_dft for energy)
# 5) Perform explicit RK3 step
# 6) Perform implicit step
# 7) Update t, step_count

v_pos_update_op = reduce(tf.group, [v_pos.assign(x) for (v_pos, x) in zip(v_pos, navier_stokes.freq_to_position_space(v_dft))])
with tf.control_dependencies([v_pos_update_op]):
    h_cfl_update_op = tf.group(h.assign(navier_stokes.compute_h_cfl(v_pos, k_cmpts, None)))
with tf.control_dependencies([h_cfl_update_op]):
    vv_dft_update_op = reduce(tf.group, [vv_dft.assign(x) for (vv_dft, x) in (vv_dft, navier_stokes.position_space_to_vv_dft(v_pos))])
# FIXME - output taps should go here
