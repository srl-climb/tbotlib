from __future__      import annotations
from typing          import Tuple, Type

from tbotlib.matrices import TransformMatrix
from .TbObject       import TbObject
from ..tools          import hyperRectangle
import numpy as np

class TbSet(TbObject):

    def hdistance(self, normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:

        pass


class TbPolytopeSet(TbSet):

    def __init__(self, vertices: np.ndarray, **kwargs) -> None:

        self.vertices       = vertices
        self.vertices_world = np.empty(self.vertices.shape)

        super().__init__(**kwargs)


    def hdistance(self, normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        
        return (self.vertices_world @ normals - offsets)
    

class TbElliptoidSet(TbSet):

    def __init__(self, a: np.ndarray, **kwargs) -> None:

        self.a = a[:, np.newaxis]  # semi major axes
        self.A = np.diag(a * a)    # helper matrix

        super().__init__(**kwargs)

    def hdistance(self, normals: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        
        # closest or farest point to plane
        e = (self.A @ ((1 / np.linalg.norm(normals * self.a, axis=0)) * normals))

        return np.hstack((np.sum(e * normals, axis=0) - offsets, np.sum(-e * normals, axis=0) - offsets))


class TbRectangleSet(TbPolytopeSet):

    def __init__(self, mins: np.ndarray, maxs: np.ndarray, **kwargs) -> None:

        self.mins = mins
        self.maxs = maxs

        vertices = hyperRectangle(mins, maxs)

        super().__init__(vertices = vertices, **kwargs)


class TbTetherForceSet(TbRectangleSet):

    def __init__(self, f_min: np.ndarray, f_max: np.ndarray, **kwargs) -> None:

        self.f_min = f_min
        self.f_max = f_max
        
        super().__init__(mins = f_min, maxs = f_max, **kwargs)


class TbRectangleWrenchSet(TbRectangleSet):

    def __init__(self, w_mins: np.ndarray, w_maxs: np.ndarray, static: bool = False, **kwargs) -> None:

        self.static = static

        super().__init__(mins = w_mins, maxs = w_maxs, **kwargs)

    def _update_transforms(self) -> None:
        
        super()._update_transforms()

        if not self.static:
            self.vertices_world[:, :3] = (self.R_world.T @ self.vertices[:, :3].T).T
            self.vertices_world[:, 3:] = (self.R_world.T @ self.vertices[:, :3].T).T

class TbElliptoidWrenchSet(TbElliptoidSet):

    def __init__(self, w: np.ndarray, **kwargs) -> None:

        super().__init__(a = w, **kwargs)