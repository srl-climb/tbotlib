from __future__ import annotations
import numpy as np

# Helper matrices for faster calculations
H0 = np.array(((1,0,0),(1,0,0),(1,0,0)))
H1 = np.array(((0,1,0),(0,1,0),(0,1,0)))
H2 = np.array(((0,0,1),(0,0,1),(0,0,1)))
I  = np.eye(3)

def planeside(a: np.ndarray, b: np.ndarray, c: np.ndarray, X: np.ndarray) -> np.ndarray:
    '''
    Determine if points X lie on one or the other side of a 3D-plane defined by a, b, c
    a: First point of the plane.
    b: Second point of the plane.
    c: Third point of the plane.
    X: Row-wise points.

    Based on: https://math.stackexchange.com/a/214194
    '''

    # Build matrices M_i those columns are b-a, c-a, x_i-a (x_i is i-th row of X)
    ba = I * (b-a) @ H0         # first column
    ca = I * (c-a) @ H1         # second column
    xa = X[:, :, None] * H2     # third columns

    M = ba + ca + xa

    # Determine on wich side of the plane a,b,c the points of X lie
    return np.linalg.det(M) > 0

