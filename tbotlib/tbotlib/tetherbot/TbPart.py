from __future__    import annotations
from .TbObject     import TbObject
from .TbGeometry   import TbGeometry
from typing        import Type

class TbPart(TbObject):

    def __init__(self, geometries: list[Type[TbGeometry]] = None, children: list[Type[TbObject]] = None, **kwargs) -> None:
        
        if geometries is None:
            geometries = []

        if children is None:
            children = []
       
        super().__init__(children = children + geometries, **kwargs)
        
        self._geometries = geometries

    @property 
    def geometries(self) -> list[Type[TbGeometry]]:

        return self._geometries

    @geometries.setter
    def geometries(self, value: list[Type[TbGeometry]]) -> None:

        #remove old geometries
        for geometry in self._geometries:
            self.remove_geometry(geometry)

        self._geometries = []
        
        for geometry in value:
            self.add_geometry(geometry)

    def remove_geometry(self, geometry: Type[TbGeometry]) -> None:

        self._geometries.remove(geometry)
        self._remove_child(geometry)

    def remove_geometries(self) -> None:

        for geometry in self._geometries:
            self._remove_child(geometry)

        self._geometries.clear()

    def add_geometry(self, geometry: Type[TbGeometry]) -> None:

        self._geometries.append(geometry)

        geometry.parent = self
