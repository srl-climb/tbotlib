from __future__     import annotations
from .TbPoint       import TbGripPoint, TbAnchorPoint, TbDockPoint, TbHoverPoint, TbMarker
from .TbPart        import TbPart
from .TbGeometry    import TbCylinder, TbSphere
from copy           import deepcopy
import numpy as np

class TbGripper(TbPart):

    def __init__(self, hoverpoint: TbHoverPoint = None, grippoint: TbGripPoint = None, anchorpoint: TbAnchorPoint = None, dockpoint: TbDockPoint = None, marker: TbMarker = None, **kwargs) -> None:

        if hoverpoint is None:
            hoverpoint = TbHoverPoint()
        
        if grippoint is None:
            grippoint = TbGripPoint()

        if anchorpoint is None:
            anchorpoint = TbAnchorPoint()

        if dockpoint is None:
            dockpoint = TbDockPoint()

        if marker is None:
            marker = TbMarker()

        super().__init__(children = [hoverpoint, grippoint, anchorpoint, dockpoint, marker], **kwargs)

        self._hoverpoint  = hoverpoint
        self._grippoint   = grippoint
        self._anchorpoint = anchorpoint
        self._dockpoint   = dockpoint
        self._marker      = marker

    @property
    def hoverpoint(self) -> TbHoverPoint:

        return self._hoverpoint

    @property
    def grippoint(self)-> TbGripPoint:

        return self._grippoint

    @property
    def anchorpoint(self) -> TbAnchorPoint:

        return self._anchorpoint

    @property
    def dockpoint(self)-> TbDockPoint:

        return self._dockpoint
    
    @property
    def marker(self) -> TbMarker:

        return self._marker

    @staticmethod
    def create(hoverpoint: np.ndarray = [0,0,0], grippoint: np.ndarray = [0,0,0], anchorpoint: np.ndarray = [0,0,0], dockpoint: np.ndarray = [0,0,0], **kwargs) -> TbGripper:
      
        hoverpoint  = TbHoverPoint(T_local = hoverpoint)
        grippoint   = TbGripPoint(T_local = grippoint)
        anchorpoint = TbAnchorPoint(T_local = anchorpoint)
        dockpoint   = TbDockPoint(T_local = dockpoint)
       
        return TbGripper(hoverpoint = hoverpoint, grippoint = grippoint, anchorpoint = anchorpoint, dockpoint = dockpoint, **kwargs)

    @staticmethod
    def example():

        geometries = [TbCylinder(T_local = [0,0,0.015], radius = 0.015, height = 0.03), TbSphere(T_local = [0,0,0.03], radius=0.02)]

        return TbGripper.create(hoverpoint = [0,0,0.1], grippoint = [0,0,0], anchorpoint = [0,0,0.03], dockpoint = [0,0,0.05], geometries = geometries)
        # hoverpoint in simulation: [0,0,0.15]
    
    @staticmethod
    def batch(k: int = 0, **kwargs) -> list[TbGripper]:

        grippers = []

        for _ in range(k): 
            grippers.append(TbGripper.create(deepcopy(**kwargs)))

        return grippers




