from __future__  import annotations
from .TbPart     import TbPart
from .TbHold     import TbHold
from .TbGeometry import TbCylinder
import numpy as np

class TbWall(TbPart):

    def __init__(self, holds: list[TbHold] = None, **kwargs) -> None:

        if holds is None:
            holds = []

        super().__init__(children=holds, **kwargs)

        self._holds = holds
        self._n     = len(holds)
        self._A     = np.empty((3, self.n))

    @property
    def holds(self) -> list[TbHold]:

        return self._holds

    @property
    def n(self) -> int:

        return self._n

    @property
    def A_world(self) -> np.ndarray:

        for i in range(self._n):
            self._A[:,i] = self._holds[i].r_world

        return self._A
    
    def get_hold(self, id: str) -> TbHold:

        if id.isnumeric():
            i = int(id)
        else:
            i = [hold.name for hold in self._holds].index(id)

        return self._holds[i]

    @staticmethod
    def example():

        C   = np.array([[0.5,0,2.15], [0.5,0.5,2.15], [0.5,-0.5,2.15], [-0.5,0.5,2.15],[-0.5,-0.5,2.15], 
                        [0.8,0,2.15], [0.8,0.5,2.15], [0.8,-0.5,2.15], [-0.2,0.5,2.15],[-0.2,-0.5,2.15]])

        holds = TbHold.batch(C, hoverpoint = [0,0,0.05],  grippoint = [0,0,0])

        for hold in holds:
            hold.add_geometry(TbCylinder(radius = 0.05, height = 0.03))
        
        return TbWall(holds = holds)


""" C   = np.array([[0.7,0.3,2.15], 
                [0.7,0.7,2.15], #1: G2
                [0.7,0.0,2.15], 
                [0.0,0.7,2.15],
                [0.0,0.0,2.15], 
                [0.0,0.3,2.15], #5
                [0.0,1.0,2.15], #6
                [0.0,1.3,2.15],
                [0.3,0.0,2.15],
                [0.3,0.3,2.15],
                [0.3,0.7,2.15], #10
                [0.3,1.0,2.15],
                [0.3,1.3,2.15],
                [0.7,1.0,2.15], #13
                [0.7,1.3,2.15],  #14: G1
                [1.0,0.0,2.15], #15
                [1.0,0.3,2.15],
                [1.0,0.7,2.15],
                [1.0,1.0,2.15],
                [1.0,1.3,2.15],  #19: G0
                [1.3,0.0,2.15],
                [1.3,0.3,2.15],
                [1.3,0.7,2.15],  #22: G4
                [1.3,1.0,2.15],
                [1.3,1.3,2.15]]) #24: G3 """