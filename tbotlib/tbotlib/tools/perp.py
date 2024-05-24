from __future__ import annotations
import numpy as np

R = np.array([[0,1],[-1,0]])

def perp(u: np.ndarray, v: np.ndarray = None) -> np.ndarray:
    '''
    Caclulate perpendicular unit vector of 2d vector u
    u: 2D vector for which to calculate perpendicular unit vector
    v: 2D vector to indicate the direction of the perpendicular unit vector, v should not be parallel to u
    '''

    u = R @ u

    if v is not None:
        if np.dot(u,v) < 0:
            u *= -1

    return u/np.linalg.norm(u)

if __name__ == '__main__':

    u = [-2,0]
    v = [-1,-1]
    print(perp(u, v))
    print(u)