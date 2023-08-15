from __future__    import annotations
from ..tetherbot   import TbObject
from .TbObjectplot import TbObjectplot
from typing        import Type

class TbNameplot(TbObjectplot):

    def __init__(self, data: Type[TbObject], **kwargs) -> None:

        super().__init__(data, **kwargs)

    def _create(self, fontsize: float = 10, namevisibility: bool = True) -> None:
 
        self._name = self._ax.text(0,0,0, self._data.name, visible = namevisibility, fontsize = fontsize)

    def update(self):
        print(self._data.T_world.r)
        self._name.set_position_3d(self._data.T_world.r)

if __name__ == '__main__':

    TbNameplot(TbObject())
