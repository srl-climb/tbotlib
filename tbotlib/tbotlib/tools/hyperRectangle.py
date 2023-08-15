from __future__ import annotations
import itertools
import numpy as np

def hyperRectangle(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    '''
    Returns the vertices of a hyper rectangle, which is defined by the boundaries a and b
    a: Vector, contains the minimum values of the rectangle for each dimension
    b: Vector, contains the maximum values of the rectangle for each dimension
    '''

    # dimension of the hyper rectangle
    m = len(a)

    # create unit hyperrectangle
    if m == 1:
        h = np.array([0,1])
    else:
        h = np.array([p for p in itertools.product([0,1], repeat=m)])
    
    # vertices of the hyperrectangle defined by the boundaries a and b
    v = (h==1)*a + (h==0)*b

    # remove duplicate vertices
    v = np.unique(v,axis=0)

    return v



if __name__ == '__main__':

    a = [0,0,0,0]
    b = [1,1,1,1]

    v = hyperRectangle(a,b)

    print(v)

    a = [1,1,-1,3]
    b = [1,1,-1,3]

    v = hyperRectangle(a,b)

    print(v)

    print(np.zeros((1,4)))