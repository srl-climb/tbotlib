from __future__ import annotations
from .TbObject import TbObject
import numpy as np

class TbCircle(TbObject):
    '''
    Circle on x-y-plane with z-rotation axis
    '''

    def __init__(self, radius = 1, **kwargs) -> None:
        
        super().__init__(**kwargs)

        self._radius = radius
        self._radial = np.array([1,0,0], dtype=float)

    @property
    def radius(self) -> float:

        return self._radius

    def distance(self, point: np.ndarray) -> float:

        # transform from world to cricle
        point = np.round(self._T_world.Rinv @ (point - self._T_world.r), 6)
        # Note: Rounding avoids short radial vectors which lead to nan during division

        # create radial vector
        if np.any(point[:2]):
            self._radial[:2] = point[:2]
        else:
            self._radial[:] = [1,0,0]

        # calculate distance
        return np.linalg.norm(point - self._radius * (self._radial / np.linalg.norm(self._radial)))      


