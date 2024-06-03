from __future__         import annotations
from .rotM              import rotM, decompose
from .NdTransformMatrix import NdTransformMatrix
import numpy as np
import quaternion as qu

class TransformMatrix(NdTransformMatrix):

    def __init__(self, a = None, b = None) -> None:
        
        super().__init__(a, b, ndim = 3)

    def _parse(self, a = None, b = None, ndim: int = None) -> np.ndarray:
        
        # add parse option for parsing [x, y, z, alpha, beta, gamma] and [x, y, z, qw, qx, qy, qz]
        if type(a) is list or type(a) is tuple or type(a) is np.ndarray:
            if len(a) == 6:
                b = rotM(a[3], a[4], a[5], order = 'xyz')
                a = a[:3]
            if len(a) == 7:
                b = qu.as_rotation_matrix(qu.from_float_array(a[3:]))
                a = a[:3]
               
        return super()._parse(a, b, ndim)  

    def _inverse(self) -> None:
        
        if self._update:
            self._update         =  False
            self._Tinv[0:3, 0:3] =  self.R.T
            self._Tinv[0:3, -1]  = -self.R.T @ self.r

    @property
    def ex(self) -> np.ndarray:

        return self._T[:3,0]

    @property
    def ey(self) -> np.ndarray:

        return self._T[:3,1]
 
    @property
    def ez(self) -> np.ndarray:

        return self._T[:3,2]
    
    @property
    def x(self) -> float:

        return self._T[0,3]
    
    @property
    def y(self) -> float:

        return self._T[1,3]
    
    @property
    def z(self) -> float:

        return self._T[2,3]
    
    @property
    def theta_x(self) -> float:

        return self.decompose()[3]
    
    @property
    def theta_y(self) -> float:

        return self.decompose()[4]
    
    @property
    def theta_z(self) -> float:

        return self.decompose()[5]
    
    @property
    def q(self) -> np.ndarray:

        return qu.as_float_array(qu.from_rotation_matrix(self.R))
    
    @q.setter
    def q(self, value: np.ndarray):

        self.R = qu.as_rotation_matrix(qu.from_float_array(value))

    def toNdTransform(self) -> NdTransformMatrix:

        return NdTransformMatrix(self._T)

    def translate(self, r: np.ndarray) -> TransformMatrix:

        self.r = self.r + r

        return self

    def rotate(self, theta_x: float, theta_y: float, theta_z: float, order: str = 'xyz') -> TransformMatrix:

        self.R = self.R @ rotM(theta_x, theta_y, theta_z, order)

        return self
    
    def rotate_around_axis(self, theta: float, axis: np.ndarray) -> TransformMatrix:
        
        axisnorm = np.linalg.norm(axis)
    
        if axisnorm > 0.000001:
            axis = axis / axisnorm
            theta = np.deg2rad(theta)

            # Rodrigues rotation
            K = np.cross(np.eye(3), axis)
            self.R = self.R @ np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        return self

    def decompose(self, order: str ='xyz') -> np.ndarray:

        v = np.hstack((self.r, np.array(decompose(self.R, order))))

        return v

    def compose(self, v: np.ndarray) -> TransformMatrix:
        
        self.r = v[:3]
        self.R = rotM(v[3], v[4], v[5], order = 'xyz')
        
        return self

    """ def toQuaternion(self) -> np.ndarray:

        return np.hstack((self.r, qu.as_float_array(qu.from_rotation_matrix(self.R)))) """
