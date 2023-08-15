from __future__ import annotations
import numpy as np

def interweave(a: np.ndarray, b: np.ndarray, axis: int = 0) -> np.ndarray:
    '''
    Returns interweaved array c of the shape [a0, b0, a1, b1, ... , an, bn]
    a:      array
    b:      array
    axis:   0: column wise operation
            1: row-wise operation
    c:      interweaved array
    '''

    if axis == 0:
        a = a.T
        b = b.T

    c = np.empty((a.shape[0] + b.shape[0], a.shape[1]), dtype=np.common_type(a,b))
    print(a.shape)
    print(b.shape)
    print(c.shape)

    c[0::2] = a
    c[1::2] = b

    if axis == 0:
        c = c.T

    return c