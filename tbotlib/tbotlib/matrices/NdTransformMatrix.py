from __future__ import annotations
import numpy as np

class NdTransformMatrix():
    
    def __init__(self, a = None, b = None, ndim: int = None) -> None:

        self._T       = self._parse(a, b, ndim)
        self._ndim    = len(self._T)-1
        self._Tinv    = np.identity(self._ndim+1)
        self._h       = np.ones((self._ndim+1, 1))
        self._update  = True

    def _parse(self, a = None, b = None, ndim: int = None) -> np.ndarray:

        '''
        Input formats
        r, R        list, tuple, np.ndarray | list, tuple, np.ndarray
        T           NdTransformMatrix, list, tuple, np.ndarray
        r, ndim:    list, tuple, np.ndarray | scalar
        R, ndim:    list, tuple, np.ndarray | scalar
        T, ndim:    list, tuple, np.ndarray | scalar
        ndim:       int
        '''

        # a transform object was passed
        if isinstance(a, NdTransformMatrix):
            
            T = a._T

        # a was passed
        elif b is None and ndim is None:
            
            if type(a) is np.ndarray:
                T = a
            
            elif type(a) is list or type(a) is tuple:
                T = np.array(a)

        # ndim was passed
        elif a is None and b is None:

            T = np.identity(ndim+1)
            
        # a and ndim were passed
        elif b is None:

            T = np.identity(ndim+1)
            
            if type(a) is list or type(a) is tuple:
                a = np.array(a)
            
            if type(a) is np.ndarray:
                
                # just translation
                if a.shape == (ndim, ): 
                    T[:ndim, -1] = a

                # just rotation
                elif a.shape == (ndim, ndim):
                    T[:ndim, :ndim] = a

                # transformation matrix
                elif a.shape == (ndim+1, ndim+1):
                    T = a

        # a and b were passed
        elif ndim is None:

            ndim            = len(a)
            T               = np.identity(ndim+1)
            T[:ndim, -1]    = a
            T[:ndim, :ndim] = b

        # a, b, ndim were passed
        else:

            T               = np.identity(ndim+1)
            T[:ndim, -1]    = a
            T[:ndim, :ndim] = b
        
        return T

    def _inverse(self) -> None:

        if self._update:
            self._Tinv   = np.linalg.inv(self._T)
            self._update = False
    
    @property
    def T(self) -> np.ndarray:

        return self._T

    @T.setter
    def T(self, value: np.ndarray) -> None:
        self._update = True
        self._T      = value

    @property
    def r(self) -> np.ndarray:

        return self._T[:self._ndim, -1]

    @r.setter
    def r(self, value: np.ndarray) -> None:
        self._update             = True
        self._T[:self._ndim, -1] = value

    @property
    def R(self) -> np.ndarray:

        return self._T[:self._ndim, :self._ndim]

    @R.setter
    def R(self, value: np.ndarray) -> None:
        self._update                      = True
        self._T[:self._ndim, :self._ndim] = value

    @property
    def ndim(self) -> int:

        return self._ndim

    @property
    def Tinv(self) -> np.ndarray:

        self._inverse()

        return self._Tinv

    @property
    def rinv(self) -> np.ndarray:

        self._inverse()

        return self._Tinv[:self._ndim, -1]

    @property
    def Rinv(self) -> np.ndarray:

        self._inverse()

        return self._Tinv[:self._ndim, :self._ndim]

    def set_r(self, value: np.ndarray) -> NdTransformMatrix:

        self.r = value

        return self

    def set_R(self, value: np.ndarray) -> NdTransformMatrix:

        self.R = value

        return self

    def set_T(self, value: np.ndarray) -> NdTransformMatrix:

        self.T = value

        return self

    def set(self, a = None, b = None, ndim: int = None) -> NdTransformMatrix:

        self.T = self._parse(a, b, ndim)

        return self

    def forward_transform(self, v: np.ndarray, axis: bool = 1, copy: bool = True) -> np.ndarray:
        '''
        Apply forward transformation to matrix/vector v
        axis:   0: column-wise, 1: row-wise
        copy:   0: apply transform to v, 1: apply transform to a copy of v
        '''

        v = np.array(v, copy = copy)
        
        if axis == 1:
            v = v.T 

        v = ((self.R @ v).T + self.r).T
        # Note: This works with both 1d and 2d arrays
        
        if axis == 1:
            v = v.T 
  
        return v 

    def inverse_transform(self, v: np.ndarray, axis: bool = 1, copy: bool = True) -> np.ndarray:
        '''
        Apply inverse transformation to matrix/vector v
        axis:   0: column-wise, 1: row-wise
        copy:   0: apply transform to v, 1: apply transform to a copy of v
        '''

        v = np.array(v, copy = copy)
        
        if axis == 1:
            v = v.T 

        v = ((self.Rinv @ v).T + self.rinv).T
        # Note: This works with both 1d and 2d arrays
        
        if axis == 1:
            v = v.T 
  
        return v 


""" if __name__ == '__main__':

    E     = np.array([[1,0,0],[0,1,0],[0,0,1]])
    r     = np.array([1,2,3])
    scale = np.array([22,11,4])

    T = NdTransformMatrix(r, E, scale, 6)
    print(T())

    v = np.array((1,6,3,5,6,7))
    print(v)
    v = T.forwardTransform(v)
    print(v)
    v = T.inverseTransform(v)
    print(v)

    v = np.array(((1,6,3,5,6,7),(1,6,3,5,6,7)))
    v = v
    print(v)
    v = T.forwardTransform(v,1)
    print(v)
    v = T.inverseTransform(v,1)
    print(v) """