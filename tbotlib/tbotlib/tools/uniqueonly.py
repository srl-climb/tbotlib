from __future__ import annotations
import numpy as np

def uniqueonly(a: np.ndarray, axis: float = 1):
    '''
    Check which rows/columns contain only unique values.
    a:      Numpy array
    axis:   0: column-wise, 1: row-wise
    '''

    if axis == 1:

        b = np.sort(a,axis=1)

        return np.all(b[:,1:] != b[:,:-1], axis=1)

    if axis == 0:

        b = np.sort(a,axis=0)

        return np.all(b[1:,:] != b[:-1,:], axis=0)

