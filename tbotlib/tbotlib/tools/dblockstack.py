from __future__ import annotations
from typing     import List
import numpy as np

def dblockstack(a: List[np.ndarray]) -> np.ndarray:
    '''
    Stack matrices along the diagonal. Empty elements are filled with zeros.
    a: List of np.arrays
    b: np.array
    '''

    # extract the shape of every array in the list
    shapes = np.empty((len(a), 2), dtype=int)
    for shape, item in zip(shapes, a):
        shape[:] = item.shape

    # Allocated stacked matrix
    b = np.zeros(np.sum(shapes, axis=0))

    # Fill stacked matrix along the diagonal with elements of a
    idc = (0, 0)
    for shape, item in zip(shapes, a):
        b[idc[0]:idc[0]+shape[0], idc[1]:idc[1]+shape[1]] = item
        idc += shape

    return b

if __name__ == '__main__':

    a = []
    a.append(np.identity(2))
    a.append(np.identity(2)*2)
    a.append(np.ones((3,2))*3)

    b = dblockstack(a)

    print(b)


