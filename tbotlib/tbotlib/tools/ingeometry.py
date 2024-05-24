from __future__ import annotations
import numpy as np

eps = np.finfo(float).eps * 10

def inrectangle(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, points: np.ndarray, mode: str = 'in') -> list[bool]:
    '''
    Test if points lie inside a rectangle defined by vertices v1, v2, v3, and v4.
    v0, v1, v2, v3:     Vertices, define anti-clockwise the rectangle
    points:             Points to test, every row is a point.
    mode:               'in': include edgepoints, 'ex': exclude edgepoints
    '''
    if mode == 'in':
        return (np.cross(v1-v0,points-v0) >= -eps) & (np.cross(v2-v1,points-v1) >= -eps) & (np.cross(v3-v2,points-v2) >= -eps) & (np.cross(v0-v3,points-v3) >= -eps)
    if mode == 'ex':
        return (np.cross(v1-v0,points-v0) > +eps) & (np.cross(v2-v1,points-v1) > +eps) & (np.cross(v3-v2,points-v2) > +eps) & (np.cross(v0-v3,points-v3) > +eps)

def inpie(c: np.ndarray, r: float, d0: np.ndarray, d1: np.ndarray, points: np.ndarray, mode: str = 'in') -> list[bool]:
    '''
    Test if points lie insed a pie.
    c:      Center point of the pie.
    d0, d1: Directions, define anti-clockwise the sides of the pie. Length doesn't matter.
    r:      Radius of the pie.
    points: Points to test, every row is a point.
    mode:   'in': include edgepoints, 'ex': exclude edgepoints
    '''
    
    if mode == 'in':
        return (np.cross(d0,points-c) >= -eps) & (np.cross(d1,points-c) <= +eps) & (np.linalg.norm(points-c, axis=1) <= r+eps)
    if mode == 'ex':
        return (np.cross(d0,points-c) > +eps) & (np.cross(d1,points-c) < -eps) & (np.linalg.norm(points-c, axis=1) < r-eps)


if __name__ == '__main__':

    c1 = np.array([0,0])
    c2 = np.array([1,0])
    c3 = np.array([1,1])
    c4 = np.array([0,1])

    points = np.array([[0,0],[0.5,0.5],[0,5]])

    print(inrectangle(c1,c2,c3,c4,points))

    c = np.array([0,0])
    r = 4
    d1 = np.array([1,0])
    d2 = np.array([0,1])

    points = np.array([[0,0],[2,2],[-0.1,-1]])
    print(inpie(c,r,d2,d1,points))