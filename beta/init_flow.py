import numpy as np

def get_sinusoid(N, dtype):
    """Returns the dft of the velocity field
       u_y = sin(2*pi*x)

    Arguments:
        N: The number of collocation points in each direction

    Returns:
        The DFT of the velocity field as a list of three numpy arrays.
    """

    v_x = np.zeros(shape=[N, N, N//2 + 1], dtype=dtype)
    v_y = np.zeros(shape=[N, N, N//2 + 1], dtype=dtype)
    v_y[1,   0, 0] = +0.5j*N*N*N
    v_y[N-1, 0, 0] = -0.5j*N*N*N
    v_z = np.zeros(shape=[N, N, N//2 + 1], dtype=dtype)

    return [v_x, v_y, v_z]
