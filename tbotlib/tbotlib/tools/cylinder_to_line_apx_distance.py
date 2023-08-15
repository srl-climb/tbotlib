from __future__ import annotations
from ..matrices import TransformMatrix
from .ang3      import ang3
import numpy as np


def cylinder_to_line_apx_distance(T: TransformMatrix, radius: float, height: float, offset: float, axis: str, p1: np.ndarray, p2: np.ndarray) -> float | float | float:
    '''
    Aproximate distance between a cylinder and a line. The distance is the rotations and translation required
    to bring the line into the cylinder. The distance is 0 if the line lies inside the cylinder.
    T: Transform of the cylinder
    axis:   rotation axis of the cylinder (options: 'x', 'y', 'z')
    radius: radius of the cylinder
    offset: offset of the cylinder along axis
    height: height of the cylinder
    p1: start point of the line
    p2: end point of the line
    '''
  
    # Make x-axis the rotation axis of the cylinder
    if axis == 'x':
        pass
    elif axis == 'y':
        T.R = T.R[:,[1,0,2]]
    elif axis == 'z':
        T.R = T.R[:,[2,1,0]]

    # base of the line 
    L = _base(p1, p2, T.r)
    # ex: points from p1 to p2
    # ez: is perpendicular to the plane spanned by p1, p2 and T.r
    # ey: perpendicular to ex and ey

    # calculate rotation required to align the x-axis of the cylinder with the line
    angle = ang3(T.R[:,0], L.R[:,0])
 
    # calculate approximate translation required to move the line into the cyliner

    # line endpoints projected onto the plane spanned by p1, p2 and T.r
    p1 = ( L.R.T @ (p1 - T.r))[:2]
    p2 = ( L.R.T @ (p2 - T.r))[:2]

    # Normals and offsets of the cylinder on the plane (x is rotation axis)
    normals = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    offsets = abs(np.array((radius,-radius,offset+0.5*height,offset-0.5*height)))
    
    # Approximate distance between the points and the cylinder 
    distance = np.max(np.vstack((p1, p2)) @ normals.T - offsets)
    # if distance is smaller/equal to zero, all points lie inside the cylinder

    if distance <= 0:
        return angle, angle , 0
    else:
        return angle + distance, angle, distance


def _base(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):

    E = np.eye(3)
    r = np.array(p1)

    # ex base vector
    E[:,0] = p2-p1

    if all(E[:,0] == 0):                # special case, (p1 = p2)
        E[:,0] = np.random.randn(3)     # random vector
    
    E[:,0] /= np.linalg.norm(E[:,0])

    # ez base vector
    E[:,2] = np.cross(p3-p1, p2-p1)
    
    if all(E[:,2] == 0):                        # special case, (p1, p2, p3 on one line)
        E[:,2]  = np.random.randn(3)             # random vector
        E[:,2] -= E[:,2].dot(E[:,0])*E[:,0]     # make it orthogonal

    E[:,2] /= np.linalg.norm(E[:,2])

    # ey base vector
    E[:,1] = np.cross(E[:,0],E[:,2])

    E[:,1] /= np.linalg.norm(E[:,1])

    return TransformMatrix(r, E)

