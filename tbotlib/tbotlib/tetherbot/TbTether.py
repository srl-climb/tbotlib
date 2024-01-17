from __future__  import annotations
from .TbPart        import TbPart
from .TbPoint    import TbAnchorPoint
import numpy as np

class TbTether(TbPart):

    def __init__(self, f_min: float = 0, f_max: float = 0, anchorpoints: list[TbAnchorPoint] = None, **kwargs) -> None:

        if anchorpoints is None:
            anchorpoints = [TbAnchorPoint(), TbAnchorPoint()]
        
        self._anchorpoints = anchorpoints[0:2]
        self._tensioned    = True
        self._f_min        = f_min
        self._f_max        = f_max

        # prevent passing T_local to TbTether by setting it to None as it is dependent on the anchorpoints
        super().__init__(T_local = None, **kwargs)

        # reset anchorpoints so that parent/child relationship can be properly established after calling super
        self.anchorpoints  = anchorpoints[0:2]
    
    @property
    def tensioned(self) -> bool:
        
        return self._tensioned

    @tensioned.setter
    def tensioned(self, value: bool) -> None:

        self._tensioned = value

    @property
    def f_min(self) -> float:

        if self._tensioned:
            return self._f_min
        else:
            return 0
        
    @f_min.setter
    def f_min(self, value: float) -> None:
    
        self._f_min = value

    @property
    def f_max(self) -> float:

        if self._tensioned:
            return self._f_max
        else:
            return 0
        
    @f_max.setter
    def f_max(self, value: float) -> None:
    
        self._f_max = value

    @property
    def anchorpoints(self) -> list[TbAnchorPoint]:

        return self._anchorpoints

    @anchorpoints.setter
    def anchorpoints(self, value: list[TbAnchorPoint]) -> None:

        self._anchorpoints = value[0:2]

        # make tether child of both anchorpoints and anchorpoint[0] the main parent of tether
        # this way self._update_transform will be called if the local transform of any anchorpoint changes
        self.parent = self._anchorpoints[0]
        self._anchorpoints[1]._add_child(self)  
        self._update_transforms()

    @property
    def vector(self) -> np.ndarray:

        return self._anchorpoints[0].r_world - self._anchorpoints[1].r_world

    @property
    def length(self) -> float:

        return np.linalg.norm(self.vector)

    def _update_transforms(self) -> None: 

        if self._fast_mode == False:

            # Point between anchorpoint 0 and anchorpoint 1
            r = 0.5 * (self._anchorpoints[0].r_world + self._anchorpoints[1].r_world)
            
            if all(self._anchorpoints[1].r_world - self._anchorpoints[0].r_world == 0):
                # Use identity matrix for R, if anchorpoints lie on top of each other
                R = np.identity(3)

            else:
                # Calculate rotation matrix
                R = self._T_local.R
                
                # z base vector (aligned with tether)
                R[:,2]  = self._anchorpoints[1].r_world - self._anchorpoints[0].r_world
                R[:,2] /= np.linalg.norm(R[:,2])
                
                # y base vector (arbitrary direciton)
                R[:,1] -= R[:,1].dot(R[:,2])*R[:,2] 
                R[:,1] /= np.linalg.norm(R[:,1])
                
                # x base vector (arbitrary direciton)
                R[:,0] = np.cross(R[:,2],R[:,1])
                R[:,0] /= np.linalg.norm(R[:,0])

            # update T_local
            if self._parent is not None:
                self._T_world.R = R
                self._T_world.r = r
                self._T_local.T = self._parent.T_world.Tinv @ self._T_world.T
            else:
                self._T_local.R = R
                self._T_local.r = r
                
            super()._update_transforms()
        
    @staticmethod
    def create(**kwargs) -> TbTether:

        return TbTether(**kwargs)

    @staticmethod
    def example() -> TbTether:

        return TbTether.create(f_min = 0, f_max = 1000)

    @staticmethod
    def batch(m: int = 0, **kwargs) -> list[TbTether]:

        tethers = []
        for _ in range(m):
            tethers.append(TbTether(**kwargs))

        return tethers