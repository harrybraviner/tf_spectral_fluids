import numpy as np

def generate_indices(shape):
    """Returns an generator over all of the indices of an ndarray
    """

    n = len(shape)
    cur = [0]*n
    while cur[0] < shape[0]:
        yield tuple(cur)
        cur[-1] += 1
        for i in range(n-1, 0, -1):
            if cur[i] >= shape[i]:
                cur[i] = 0
                cur[i-1] += 1
            else:
                break
    return

