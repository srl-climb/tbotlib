from __future__  import annotations
from .TbPoint    import TbHoverPoint, TbGripPoint
from .TbGeometry import TbCylinder
from .TbPart        import TbPart
from copy        import deepcopy
import numpy as np


class TbHold(TbPart):

    def __init__(self, hoverpoint: TbHoverPoint = None, grippoint: TbGripPoint = None, **kwargs) -> None:

        if hoverpoint is None:
            hoverpoint = TbHoverPoint()
        
        if grippoint is None:
            grippoint = TbGripPoint()

        super().__init__(children = [hoverpoint, grippoint], **kwargs)

        self._hoverpoint = hoverpoint
        self._grippoint  = grippoint

    @property
    def hoverpoint(self) -> TbHoverPoint:

        return self._hoverpoint

    @property
    def grippoint(self)-> TbGripPoint:

        return self._grippoint

    @staticmethod
    def create(hoverpoint: np.ndarray = [0,0,0.05], grippoint: np.ndarray = [0,0,0], **kwargs) -> TbHold:

        hoverpoint = TbHoverPoint(T_local = hoverpoint)
        grippoint  = TbGripPoint(T_local = grippoint)

        return TbHold(hoverpoint = hoverpoint, grippoint = grippoint, **kwargs)

    @staticmethod
    def example() -> TbHold:

        geometries  = [TbCylinder(radius = 0.05, height = 0.03)]

        return TbHold.create(hoverpoint = [0,0,0.05], grippoint = [0,0,0], geometries = geometries)

    @staticmethod
    def batch(C: np.ndarray, **kwargs) -> list[TbHold]:

        holds = []

        for c in C: # C contains row wise points
            holds.append(TbHold.create(T_local = c, **deepcopy(kwargs)))

        return holds