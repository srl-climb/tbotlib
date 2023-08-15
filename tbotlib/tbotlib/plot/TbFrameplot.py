from __future__    import annotations
from ..tetherbot   import TbObject
from .TbObjectplot import TbObjectplot
from typing        import Type
import numpy as np

class TbFrameplot(TbObjectplot):

    def __init__(self, data: Type[TbObject], **kwargs) -> None:
        
        super().__init__(data, **kwargs)

    def _create(self, framevisibility: bool = True, framesize: float = 0.01, framecolor: list[str] = ['r','g','b'], **kwargs) -> None:
        
        super()._create(**kwargs)

        self._framesize = framesize
        self._E         = np.eye(3)

        self._xline = self._ax.plot([0,1], [0,0], [0,0], framecolor[0])[0]
        self._yline = self._ax.plot([0,0], [0,1], [0,0], framecolor[1])[0]
        self._zline = self._ax.plot([0,0], [0,0], [0,1], framecolor[2])[0]

        self._xline.set_visible(framevisibility)
        self._yline.set_visible(framevisibility)
        self._zline.set_visible(framevisibility)

    def update(self):
        
        T = self._data.T_world
        E = T.R @ self._E * self._framesize

        self._xline.set_xdata([T.r[0], E[0,0]+T.r[0]])
        self._yline.set_xdata([T.r[0], E[0,1]+T.r[0]])
        self._zline.set_xdata([T.r[0], E[0,2]+T.r[0]])

        self._xline.set_ydata([T.r[1], E[1,0]+T.r[1]])
        self._yline.set_ydata([T.r[1], E[1,1]+T.r[1]])
        self._zline.set_ydata([T.r[1], E[1,2]+T.r[1]])

        self._xline.set_3d_properties([T.r[2], E[2,0]+T.r[2]])
        self._yline.set_3d_properties([T.r[2], E[2,1]+T.r[2]])
        self._zline.set_3d_properties([T.r[2], E[2,2]+T.r[2]])