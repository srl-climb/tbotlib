from __future__ import annotations
from ..matrices import TransformMatrix, NdTransformMatrix, rotM
from .ang3      import ang3
from typing     import Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..tetherbot import TbTetherbot

def basefit(points: np.ndarray, axis: int = 0, output_format: int = 0) -> Tuple[np.ndarray, np.ndarray] | TransformMatrix | NdTransformMatrix:

    '''
    Fit a coordinate system to a set of points.
    points: numpy array
    axis:           0: column wise (each column of points is a point)
                    1: row wise (each row of points is a point)
    output_format:  0:   Returns the support vector and basis (each column is a basis vector) of the coordinate system  
                    1/2: Retruns transformation of the coordinate system
    Note: Axis has no effect if only a single point was passed
    Note: For n<=3, the first point is used as the support vector
    '''

    # set axis to 1 if only a single point was passed
    if points.ndim == 1:
        axis = 1

    # prepare base and support vector
    E = np.eye(3)
    r = np.zeros(3)

    # ensure points is 2 dimensional
    points = np.atleast_2d(points)

    # Change to column wise form (each column is a point)
    if axis == 1:
        points = points.T

    # Remove duplicate points, without sorting of unique
    points = points[:, np.sort(np.unique(points, axis=1, return_index=True)[1])]

    # Check if points has the right shape (3xn)
    if axis == 0:
        assert points.shape[0] == 3, 'Points do not have the expected shape of 3xn (row x col)'
    elif axis == 1:
        assert points.shape[0] == 3, 'Points do not have the expected shape of nx3 (row x col)'

    # Calculate the base and support vector

    # A single point was passed
    if points.shape[1] == 1:

        r = points[:,0]

    # Two points were passed
    elif points.shape[1] == 2:
        
        r = points[:,0]

        # first base vector
        E[:,0] = points[:,1] - points[:,0]
        E[:,0] /= np.linalg.norm(E[:,0])

        # second base vector
        E[:,1] = np.random.randn(3)             # random vector
        E[:,1] -= E[:,1].dot(E[:,0])*E[:,0]     # make it orthogonal
        E[:,1] /= np.linalg.norm(E[:,1])

        # third base vector
        E[:,2] = np.cross(E[:,0],E[:,1])
        E[:,2] /= np.linalg.norm(E[:,2])


    # Three points were passed
    elif points.shape[1] == 3:

        r = points[:,0]

        # first base vector
        E[:,0] = points[:,1] - points[:,0]
        E[:,0] /= np.linalg.norm(E[:,0])

        # second base vector
        E[:,1] = points[:,1] - points[:,2]
        E[:,1] /= np.linalg.norm(E[:,1])

        # third base vector
        E[:,2] = np.cross(E[:,0],E[:,1])
        E[:,2] /= np.linalg.norm(E[:,2])

    # More than three points were passed
    elif points.shape[1] >= 3:

        #code drawn from https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
        r = np.mean(points, axis=1, keepdims=True)

        # base vectors
        E = np.linalg.svd((points - r))[0]
        r = r[:,0]

    # Output results
    if output_format == 0:
        
        return r, E
    
    elif output_format == 1:

        return TransformMatrix(r, E)

    elif output_format == 2:

        return NdTransformMatrix(r, E)
    
    else:

        pass

def tbbasefit(tbot: TbTetherbot, output_format: int = 0 ):
    '''
    Fit a coordinate system to a tetherbot
    points: numpy array
    output_format:  0:   Returns the support vector and basis (each column is a basis vector) of the coordinate system  
                    1/2: Retruns transformation of the coordinate system
    '''

    # temporary transformation matrix
    T_temp = basefit(tbot.A_world[:, tbot.tensioned], axis = 0, output_format = 1)
    # basefit returns transformation based on a singular-value-decomposition, hence the axes might be flipped/mirrored

    # flip z axis in the correct direction using one of the grippers z axes as a reference direction
    if np.dot(T_temp.ez, tbot.grippers[0].T_world.ez) < 0:
        T_temp._T[:3,2] *= -1

    # flip y axis to ensure right handed system
    if np.dot(np.cross(T_temp.ez, T_temp.ex), T_temp.ey) < 0:
        T_temp._T[:3,1] *= -1

    # find ex, ey where cable lengths are the shortest
    l_min = np.inf
    R_min = np.eye(3)

    for i in range(4):
        R = T_temp.R @ rotM(0, 0, 90*i, 'xyz')

        # sum of all tether lengths
        l = np.sum(np.linalg.norm(tbot.A_world[:, tbot.tensioned] - (T_temp.r[:, None] + R @ tbot.B_local[ :,tbot.tensioned]), axis=0))
        
        if l < l_min:
            l_min = l
            R_min[:] = R[:]    
    T_temp._T[:3,:3] = R_min

    return _format_output(T_temp.r, T_temp.R, output_format)


def _format_output(r: np.ndarray, E: np.ndarray, output_format: int):

    if output_format == 0:
        
        return r, E
    
    elif output_format == 1:

        return TransformMatrix(r, E)

    elif output_format == 2:

        return NdTransformMatrix(r, E)
    
    else:

        pass

    return