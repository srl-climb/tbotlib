from __future__     import annotations
from ..matrices     import TransformMatrix
from .TbArm         import TbArm, TbRPPArm
from .TbPart        import TbPart
from .TbPoint       import TbAnchorPoint, TbCamera, TbDepthsensor
from .TbGeometry    import TbAlphashape, TbCylinder, TbGeometry, TbBox
from typing         import Type, Union
import numpy as np

class TbPlatform(TbPart):

    def __init__(self, arm: TbArm = None, anchorpoints: list[TbAnchorPoint] = None, cameras: list[TbCamera] = None, depthsensor: TbDepthsensor = None, **kwargs) -> None:
        
        if arm is None:
            arm = TbArm()

        if anchorpoints is None:
            anchorpoints = []
        
        if cameras is None:
            cameras = []
        
        if depthsensor is None:
            depthsensor = []
        else:
            depthsensor = [depthsensor]
       
        super().__init__(children = [arm] + anchorpoints + cameras + depthsensor, **kwargs)
        
        self._arm          = arm
        self._anchorpoints = anchorpoints
        self._cameras      = cameras
        self._depthsensor  = depthsensor
        self._m            = len(anchorpoints)
        self._B            = np.empty((3, self._m))

    @property
    def arm(self) -> Union[TbArm, TbRPPArm]:
        
        return self._arm

    @property
    def anchorpoints(self) -> list[TbAnchorPoint]:

        return self._anchorpoints
    
    @property
    def cameras(self) -> list[TbCamera]:

        return self._cameras
    
    @property
    def depthsensor(self) -> list[TbDepthsensor]:

        return self._depthsensor

    @property
    def m(self) -> int:

        return self._m

    @property
    def B_world(self) -> np.ndarray:

        for i in range(self._m):
            self._B[:,i] = self.anchorpoints[i].r_world

        return self._B

    @staticmethod
    def create(B: np.ndarray, arm: TbArm = None, geometries: list[Type[TbGeometry]] = [], **kwargs) -> TbPlatform:
        
        if arm is None:
            arm = TbArm(links=[])
            
        anchorpoints = []
        for b in B: # B contains row wise points
            anchorpoints.append(TbAnchorPoint(T_local = b))

        return TbPlatform(arm = arm, anchorpoints = anchorpoints, geometries=geometries, **kwargs)

    @staticmethod
    def example() -> TbPlatform:

        arm = TbRPPArm.example()

        B = np.array([[ 0.2, 0,    0.05],
                      [ 0.2, 0,   -0.05],
                      [ 0.2, 0.15, 0.05],
                      [ 0.2, 0.15,-0.05],
                      [-0.2, 0.15, 0.05],
                      [-0.2, 0.15,-0.05],
                      [-0.2,-0.15, 0.05],
                      [-0.2,-0.15,-0.05],
                      [ 0.2,-0.15, 0.05],
                      [ 0.2,-0.15,-0.05]])
        
        geometries = [TbAlphashape(points=B, alpha=0.01), TbCylinder(radius=0.05, height=0.02, T_local=[0,0,0.06])]
        geometries = [TbCylinder(radius=0.05, height=0.02, T_local=[0,0,0.06]), TbBox([0.4,0.3,0.1], T_local = TransformMatrix([-0.2,-0.15,-0.05]))]

        return TbPlatform.create(B, arm, geometries, T_local = [0,0,2.15])