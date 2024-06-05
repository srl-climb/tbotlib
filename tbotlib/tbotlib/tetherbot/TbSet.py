from __future__      import annotations
from typing          import Tuple, Type, Callable, TYPE_CHECKING, Union
from scipy.spatial   import HalfspaceIntersection
from tbotlib.matrices import TransformMatrix
from .TbObject        import TbObject
from ..tools          import hyperRectangle, ang3
from ..models         import GripperForceModel
import numpy as np


if TYPE_CHECKING:
    from .TbTetherbot import TbTetherbot

class TbSet(TbObject):

    @staticmethod
    def _parse_to_array(x: Union[float, list, np.ndarray], n: int) -> np.ndarray:

        if np.isscalar(x):
            arr = np.ones(n) * x
        else:
            arr = np.array(x)

        return arr


class TbTetherForceSet(TbSet):

    # helper matrix for calculating the cross product
    _H = np.array([[[ 0,  0,  0],
                    [ 0,  0,  1],
                    [ 0, -1,  0]],
                   [[ 0,  0, -1],
                    [ 0,  0,  0],
                    [ 1,  0,  0]],
                   [[ 0,  1,  0],
                    [-1,  0,  0],
                    [ 0,  0,  0]]])

    def __init__(self, f_min: Union[float, list, np.ndarray], f_max: Union[float, list, np.ndarray], parent: TbTetherbot = None, **kwargs) -> None:

        self._tetherbot = parent
        self._f_min = self._parse_to_array(f_min, self._tetherbot.m)
        self._f_max = self._parse_to_array(f_max, self._tetherbot.m)
        
        self._update_cache()
        
        super().__init__(parent = parent, **kwargs)
    
    def _update_cache(self) -> None:
        
        self._tensioned = self._tetherbot.tensioned.copy()
        self._m = np.sum(self._tensioned)
        self._vertices = hyperRectangle(self.f_min(), self.f_max())
        self._halfspaces = np.zeros((2*self._m, self._m+1))    
        self._halfspaces[:self._m, :-1] = -np.eye(self._m) 
        self._halfspaces[:self._m, -1] = self.f_min()
        self._halfspaces[self._m:2*self._m, :-1] =  np.eye(self._m) 
        self._halfspaces[self._m:2*self._m, -1] = -self.f_max()

    def _check_cache(func):
        
        def wrapper(self: TbTetherForceSet) -> np.ndarray:
            if not np.all(self._tensioned == self._tetherbot.tensioned):
                self._update_cache()
            return func(self)
        return wrapper
            
    def f_min(self) -> np.ndarray:

        return self._f_min[self._tetherbot.tensioned]
    
    def f_max(self) -> np.ndarray:

        return self._f_max[self._tetherbot.tensioned]
    
    def AT(self) -> np.ndarray:

        L = self._tetherbot.L
        B = self._tetherbot.B_world
        U = L / np.linalg.norm(L, axis=0)
            
        AT = np.zeros((L.shape[0]*2 , L.shape[1]))
        AT[:3,:] = U
        AT[3:,:] = np.einsum('ijk,ja,ka->ia', self._H, B, U) # cross product (faster than np.cross)

        return AT[:, self._tetherbot._tensioned]

    @_check_cache
    def vertices(self) -> np.ndarray:

        return self._vertices

    @_check_cache
    def halfspaces(self) -> np.ndarray:

        return self._halfspaces
    
    def m(self) -> int:

        return np.sum(self._tetherbot.tensioned)
    
    def in_set(self, f: np.ndarray) -> bool:
        
        return np.all(f <= self.f_max()) and np.all(f >= self.f_min())


class TbTetherGripperForceSet(TbTetherForceSet):

    def __init__(self, f_min: np.ndarray, f_max: np.ndarray, gfm: GripperForceModel, parent: TbTetherbot = None, **kwargs) -> None:

        self.gfm    = gfm

        super().__init__(f_min, f_max, parent, **kwargs)
 
    def _update_cache(self) -> None:
  
        self._tensioned = self._tetherbot.tensioned.copy()
        self._m = np.sum(self._tensioned)
        self._loaded = np.zeros(self._tetherbot.k, dtype=bool)
        self._loaded[list(set(self._tetherbot.mapping.a_to_b[self._tensioned, 0]))] = True
        self._k = np.sum(self._loaded)

        self._G_max = np.zeros((self._tetherbot._k, self._tetherbot._m))
        for i, j in zip(self._tetherbot.mapping.a_to_b[:, 0], range(self._tetherbot._m)):
            theta = ang3(self._tetherbot.grippers[i].T_world.ez, self._tetherbot.L[:, j])
            self._G_max[i, j] = 1 / self.gfm.eval(theta)

        self._halfspaces = np.zeros((2*self._m + self._k, self._m+1))    
        self._halfspaces[:self._m, :-1] = -np.eye(self._m) 
        self._halfspaces[:self._m, -1] = self.f_min()
        self._halfspaces[self._m:2*self._m, :-1] =  np.eye(self._m) 
        self._halfspaces[self._m:2*self._m, -1] = -self.f_max()
        self._halfspaces[2*self._m:, -1] = -np.ones(self._k)
        self._halfspaces[2*self._m:, :-1] = self._G_max[self._loaded, :][:, self._tensioned]

        self._interior_point = self.f_min() + 0.0000001
        
        self._vertices = HalfspaceIntersection(self._halfspaces, self._interior_point).intersections

    def in_set(self, f: np.ndarray) -> bool:
        
        if super().in_set(f):
            return np.all(self._G_max[self._loaded, :][:, self._tensioned] @ f <= 1)
        
        return False


class TbWrenchSet(TbSet):

    def hdistance(self, normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:

        pass


class TbPolytopeWrenchSet(TbWrenchSet):

    def __init__(self, vertices: np.ndarray, **kwargs) -> None:

        self.vertices       = vertices
        self.vertices_world = np.empty(self.vertices.shape)

        super().__init__(**kwargs)

    def _update_transforms(self) -> None:
        
        super()._update_transforms()

        self.vertices_world[:, :3] = (self.R_world @ self.vertices[:, :3].T).T
        self.vertices_world[:, 3:] = (self.R_world @ self.vertices[:, :3].T).T

    def hdistance(self, normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        
        return (self.vertices_world @ normals - offsets)
    

class TbRectangleWrenchSet(TbPolytopeWrenchSet):

    def __init__(self, w_mins: np.ndarray, w_maxs: np.ndarray, **kwargs) -> None:

        self.w_mins = self._parse_to_array(w_mins, 6)
        self.w_maxs = self._parse_to_array(w_maxs, 6)

        vertices = hyperRectangle(self.w_mins, self.w_maxs)

        super().__init__(vertices = vertices, **kwargs)

    
class TbElliptoidWrenchSet(TbWrenchSet):

    def __init__(self, w: np.ndarray, **kwargs) -> None:

        self.w = self._parse_to_array(w, 6)

        # helper variables
        self._a = self.w[:, np.newaxis]         # 2x1 array o semi major axes
        self._A = np.diag(self.w * self.w)    # diagonal matrix of semi major axes

        super().__init__(**kwargs)

    def hdistance(self, normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:

        # closest or farest point to plane
        e = (self._A @ ((1 / np.linalg.norm(normals * self._a, axis=0)) * normals))

        return np.hstack((np.sum(e * normals, axis=0) - offsets, np.sum(-e * normals, axis=0) - offsets))