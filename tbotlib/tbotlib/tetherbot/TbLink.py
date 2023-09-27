from __future__     import annotations
from ..matrices     import DenavitHartenberg
from ..tools        import between
from .TbPart        import TbPart
from .TbPoint       import TbDockPoint
from abc            import ABC


class TbLink(TbPart, ABC):

    def __init__(self, phi: float = 0, alpha: float = 0, a: float = 0, d: float = 0, q0: float = 0, qlim: list[float] = None, dockpoint: TbDockPoint = None, **kwargs) -> None:

        if qlim is None:
            qlim = [0,1]
        if dockpoint is None:
            dockpoint = TbDockPoint()

        self._dhp       = DenavitHartenberg(phi, alpha, a, d)
        self._q0        = between(q0, qlim)[1]
        self._qlim      = qlim
        self._dockpoint = dockpoint

        super().__init__(children = [dockpoint], T_local = None, **kwargs)

    @property
    def q(self) -> float:
        
        return 0 

    @property
    def dhp(self) -> DenavitHartenberg:

        return self._dhp

    @dhp.setter
    def dhp(self, value: DenavitHartenberg) -> None:

        self._dhp = value

    @property
    def q0(self) -> float:
        
        return self._q0

    @q0.setter
    def q0(self, value: float) -> None:

        self._q0 = between(value, self.qlim)[1]  

    @property
    def qlim(self) -> list[float]:

        return self._qlim

    @qlim.setter
    def qlim(self, value: list[float]) -> None:

        self._qlim = value

    @property
    def dockpoint(self) -> TbDockPoint:

        return self._dockpoint

    def _update_transforms(self):

        self._T_local = self._dhp.toTransformMatrix()
        
        super()._update_transforms()


class TbRevoluteLink(TbLink):

    def __init__(self, q0: float = 0, alpha: float = 0, a: float = 0, d: float = 0, **kwargs) -> None:

        super().__init__(phi = q0, alpha = alpha, a = a, d = d, **kwargs)

    @property
    def q(self) -> float:
        
        return self._dhp.phi

    @q.setter
    def q(self, value: float) -> None:

        self._dhp.phi = between(value, self.qlim)[1]    
        self._update_transforms()


class TbPrismaticLink(TbLink):

    def __init__(self, phi: float = 0, alpha: float = 0, a: float = 0, q0: float = 0, **kwargs) -> None:

        super().__init__(phi = phi, alpha = alpha, a = a, d = q0, **kwargs)

    @property
    def q(self) -> float:
        
        return self._dhp.d

    @q.setter
    def q(self, value: float) -> None:

        self._dhp.d = between(value, self.qlim)[1]
        self._update_transforms()