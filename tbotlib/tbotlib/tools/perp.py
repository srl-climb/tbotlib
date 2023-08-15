from __future__ import annotations
import numpy as np

R = np.array([[0,1],[-1,0]])

def perp(u: np.ndarray) -> np.ndarray:
    '''
    Caclulate perpendicular unit vector of 2d vector u
    u: 2D vector
    '''

    u = R @ u

    return u/np.linalg.norm(u)

if __name__ == '__main__':

    u = [-2,0]

    print(perp(u))
    print(u)