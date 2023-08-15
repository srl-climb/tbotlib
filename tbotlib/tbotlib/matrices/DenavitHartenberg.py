from __future__       import annotations
from math             import cos, sin
from .TransformMatrix import TransformMatrix
import numpy as np

class DenavitHartenberg():

    def __init__(self, phi: float, alpha: float, a: float, d: float, modified: bool = False) -> None:

        self._T        = np.identity(4)
        self._phi      = phi       
        self._alpha    = alpha
        self._a        = a         
        self._d        = d       
        self._modified = modified
        self._calculate()

    def __call__(self, phi: float, alpha: float, a: float, d: float) -> np.ndarray:
        
        self._phi   = phi
        self._alpha = alpha
        self._a     = a
        self._d     = d
        self._calculate()

        return self._T

    @property
    def T(self) -> np.ndarray:

        return self._T

    @property 
    def phi(self) -> float:

        return self._phi

    @phi.setter
    def phi(self, value: float) -> None:

        self._phi = value
        self._calculate()

    @property 
    def alpha(self) -> float:

        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:

        self._alpha = value
        self._calculate()

    @property 
    def a(self) -> float:

        return self._a

    @a.setter
    def a(self, value: float) -> None:

        self._a = value
        self._calculate()

    @property 
    def d(self) -> float:

        return self._d

    @d.setter
    def d(self, value: float) -> None:

        self._d = value
        self._calculate()


    def _calculate(self) -> None:

        if self._modified:
            
            self._T = np.array([[cos(self._phi),                 -sin(self._phi),                    0,                 self._a                 ],
                                [sin(self._phi)*cos(self._alpha),  cos(self._phi)*cos(self._alpha), -sin(self._alpha), -self._d*sin(self._alpha)],
                                [sin(self._phi)*sin(self._alpha),  cos(self._phi)*sin(self._alpha),  cos(self._alpha),  self._d*cos(self._alpha)],
                                [0,                                0,                                0,                 1                       ]])

        else:  

            self._T = np.array([[cos(self._phi), -sin(self._phi)*cos(self._alpha),  sin(self._phi)*sin(self._alpha), self._a*cos(self._phi)],
                                [sin(self._phi),  cos(self._phi)*cos(self._alpha), -cos(self._phi)*sin(self._alpha), self._a*sin(self._phi)],
                                [0,               sin(self._alpha),                 cos(self._alpha),                self._d               ],
                                [0,               0,                                0,                               1                     ]])

    def toTransformMatrix(self) -> TransformMatrix:

        return TransformMatrix(self._T)
 

        
        