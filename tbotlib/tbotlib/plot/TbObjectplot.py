from __future__  import annotations
from ..tetherbot import TbObject
from .Plot3d     import Plot3d
from typing      import Type


class TbObjectplot(Plot3d):

    def __init__(self, data: Type[TbObject], **kwargs) -> None:
        
        self._data = data

        super().__init__(**kwargs)

    @property
    def data(self) -> Type[TbObject]:

        return self._data

    @data.setter
    def data(self, value: Type[TbObject]) -> None:

        self._data = value
        self.update()