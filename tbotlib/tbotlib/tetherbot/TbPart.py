from __future__    import annotations
from .TbObject     import TbObject
from .TbGeometry   import TbGeometry
from .TbCollidable import TbCollidable
from typing        import Type

class TbPart(TbObject):

    def __init__(self, geometries: list[Type[TbGeometry]] = None, collidables: list[Type[TbCollidable]] = None, children: list[Type[TbObject]] = None, **kwargs) -> None:
        
        if geometries is None:
            geometries = []

        if children is None:
            children = []

        if collidables is None:
            collidables = []
       
        super().__init__(children = children + geometries + collidables, **kwargs)
        
        self._geometries = geometries
        self._collidables = collidables

    @property 
    def geometries(self) -> list[Type[TbGeometry]]:

        return self._geometries
    
    @property 
    def collidables(self) -> list[Type[TbCollidable]]:

        return self._collidables

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

    def remove_collidable(self, collidable: Type[TbCollidable]) -> None:

        self._collidables.remove(collidable)
        self._remove_child(collidable)

    def remove_collidables(self) -> None:

        for collidable in self._collidables:
            self._remove_child(collidable)

        self._collidables.clear()

    def add_collidable(self, collidable: Type[TbCollidable]) -> None:

        self._collidables.append(collidable)

        collidable.parent = self
