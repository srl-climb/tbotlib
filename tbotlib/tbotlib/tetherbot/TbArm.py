from __future__     import annotations
from ..tools        import between
from ..matrices     import TransformMatrix
from .TbLink        import TbLink, TbPrismaticLink, TbRevoluteLink
from .TbGeometry    import TbCylinder
from .TbCircle      import TbCircle
from .TbPart        import TbPart
from typing         import Type
from math           import pi, atan2, sqrt
import numpy as np


class TbArm(TbPart):

    def __init__(self, links: list[Type[TbLink]] = None, **kwargs) -> None:

        if links is None:
            links = []
        
        super().__init__(**kwargs)

        self._dof       = len(links)
        self._links     = links

        # automatically set up parent child relationship between links and the arm
        for link, parent in zip(links, [self] + links):
            link.parent = parent

    @property
    def links(self) -> list[TbLink]:
        
        return self._links

    @property
    def dof(self) -> int:

        return self._dof

    @property
    def q0s(self) -> np.ndarray:

        return np.array([link.q0 for link in self.links])
    
    @property
    def qs(self) -> np.ndarray:
        
        return np.array([link.q for link in self.links])

    @qs.setter
    def qs(self, values: np.ndarray) -> None:

        for value, link in zip(values, self.links):
            link.q = value

    @property
    def base(self) -> TransformMatrix:

        return self.T_world   

    @property
    def qlims(self) -> np.ndarray:

        return np.array([link.qlim for link in self.links])

    def fwk(self, qs: np.ndarray = None) -> TransformMatrix:

        T = self.base.T

        if qs is None:
            qs = self.qs

        for q, link in zip(qs, self.links):
            link.q = q
            T = T @ link.dhp.T 

        return TransformMatrix(T) 

    def ivk(self, T: TransformMatrix) -> None:

        pass

    def valid(self, qs: np.ndarray = None) -> bool:

        if qs is None:
            qs = self.qs

        for link, q in zip(self.links, qs):
            if between(q, link.qlim)[0] == 0:                
                return False

        return True

    def reachable(self, T: TransformMatrix) -> bool:

        return self.valid(self.ivk(T))


    @property
    def dhp(self) -> np.ndarray:

        return np.array([[link.dhp.phi, link.dhp.alpha, link.dhp.a, link.dhp.d] for link in self.links])


class TbRPPArm(TbArm):

    '''
    Revolute-Prismatic-Prismatic-Robot arm with DH-parameters as follows:
        phi     alpha   a   d
    1   phi_1   -90     0   d_1
    2   0       -90     0   d_2
    3   0       0       0   d_3
    '''

    def __init__(self, links: list[Type[TbLink]] = None, **kwargs) -> None:
        
        super().__init__(links = links, **kwargs)

        assert (self.dhp[0,1] == -pi/2 and self.dhp[1,1] == -pi/2 and 
                self.dhp[0,2] == 0 and self.dhp[1,0] == 0 and self.dhp[1,2] == 0  and 
                self.dhp[2,0] == 0 and self.dhp[2,1] == 0 and self.dhp[2,2]== 0), 'Incompatible DH-parameters!'

        # parameters of the cylindrical workspace
        self._workspace_radius = self.links[1].qlim[1]
        self._workspace_height = self.links[-1].qlim[1] - self.links[-1].qlim[0]
        self._workspace_offset = self.dhp[0,3] - self.links[-1].qlim[0] - 0.5*self._workspace_height #offset of the center from the base
        self._workspace_center = TbCircle(parent = self, T_local = TransformMatrix([0,0, self._workspace_offset]), radius = self._workspace_radius)

    @property
    def workspace_radius(self) -> float:

        return self._workspace_radius

    @property
    def workspace_height(self) -> float:

        return self._workspace_height

    @ property
    def workspace_offset(self) -> float:

        return self._workspace_offset

    @ property
    def workspace_center(self) -> TbCircle:

        return self._workspace_center

    def ivk(self, T: TransformMatrix) -> np.ndarray:

        r =  (self.base.Tinv @ T.T)[:3,3] 

        q0 = -atan2(r[0], r[1]) 
        q1 = sqrt(r[0]**2+r[1]**2)
        q2 = self.links[0].dhp.d - r[2] 
        qs = [q0, q1, q2]

        return qs

    @staticmethod
    def example() -> TbRPPArm:

        geometries = [TbCylinder(radius=0.050, height=0.05, T_local=[0,0,0,90,0,0]), 
                      TbCylinder(radius=0.016, height=0.30, T_local=[0,0,0.15,0,0,0])]
        link_1 = TbRevoluteLink(q0=0, alpha=-pi/2, a=0, d=0.095, qlim=[-pi, pi], geometries=geometries)

        geometries = [TbCylinder(radius=0.014, height=1.200, T_local=[0,0.616,0,90,0,0]), 
                      TbCylinder(radius=0.014, height=0.016, T_local=[0,0.007,0,90,0,0]),
                      TbCylinder(radius=0.016, height=0.050, T_local=[0,0,0,0,0,0])]

        link_2 = TbPrismaticLink(phi=0, alpha=-pi/2, a=0, q0=0.314, qlim=[0.314,1.414], geometries=geometries)

        geometries = [TbCylinder(radius=0.014, height=0.2, T_local=[0,0,-0.1,0,0,0])]
        link_3 = TbPrismaticLink(phi=0, alpha=0, a=0, q0=0.04, qlim=[0,0.3], geometries=geometries)
        
        return TbRPPArm(T_local=[0,0,0,0,0,0], links=[link_1, link_2, link_3]) #2.15
