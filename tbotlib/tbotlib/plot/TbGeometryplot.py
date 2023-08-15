from __future__    import annotations
from ..tetherbot   import TbGeometry
from .TbObjectplot import TbObjectplot
from typing        import Type

class TbGeometryplot(TbObjectplot):

    def __init__(self, data: Type[TbGeometry], **kwargs) -> None:
        
        super().__init__(data, **kwargs)

    def _create(self, geometryvisibility: bool = True, geometryalpha = 1, geometrycolor = 'b') -> None:
        
        self._surf = self._ax.plot_trisurf(*zip(*self._data.vertices), triangles = self._data.faces, alpha = geometryalpha, color = geometrycolor, visible = geometryvisibility)

    def update(self) -> None:

        self._surf.set_verts(self._data.vertices_world[self._data.faces.flatten(),:])


    """ if __name__ == '__main__':
    
    from tetherbot          import TbCylinder
    from tetherbot          import TbPoint
    from tools              import Tricylinder
    from core import show

    geometry = TbCylinder()
    TbGeometryplot(geometry)
    show() """

    """ point_1 = TbPoint(T_local=[1,0,0])
    point_2 = TbPoint()
    cylinder = Tb2PCylinder(point_1=point_1, point_2=point_2, radius=0.01)
    TbGeometryplot(cylinder)
    show() """