from __future__     import annotations
from ..tetherbot    import TbObject
from .TbObjectplot  import TbObjectplot
from typing         import Type
import numpy as np


class TbPointplot(TbObjectplot):

    def __init__(self, data: Type[TbObject], **kwargs) -> None:

        super().__init__(data, **kwargs)

    def _create(self, markersize: float = 1, marker: str = 'o', markercolor = 'r', markervisibility: bool = True) -> None:

        self._point = self._ax.scatter(0, 0, 0, marker = marker, s = markersize*50, c = markercolor, visible = markervisibility)

    def update(self):

        self._point.set_offsets(np.atleast_2d(self._data.r_world[0:2]))
        self._point.set_3d_properties(self._data.r_world[2], zdir='z')