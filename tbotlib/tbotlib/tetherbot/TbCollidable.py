from __future__ import annotations
from .TbObject      import TbObject
import open3d as o3d
import numpy  as np
from copy import deepcopy

class TbCollidable(TbObject):

    def __init__(self, geometry: o3d.geometry.LineSet = None, color = [1, 0.706, 0], **kwargs) -> None:

        self._geometry     = geometry
        self._points       = deepcopy(np.asarray(self._geometry.points))
        self._lines        = deepcopy(np.asarray(self._geometry.lines))
        self._points_world = np.empty(self._points.shape)
        self._color        = color
        self._geometry.paint_uniform_color(color)
        
        TbObject.__init__(self, **kwargs)

    def __getstate__(self):
        '''Called when serializing'''

        # copy to avoid modifying the object
        state = self.__dict__.copy()
        # remove unpickable entries (open3d objects)
        del state["_geometry"]

        return state

    def __setstate__(self, state: dict):
        '''Called when deserializing'''

        # restore original pickable state
        self.__dict__.update(state)
  
        # restore unpickable entries
        self._geometry = o3d.geometry.LineSet(o3d.utility.Vector3dVector(self._points), o3d.utility.Vector2iVector(self._lines))
        self._geometry.paint_uniform_color(self._color)

    @property
    def geometry(self) -> o3d.geometry.LineSet:

        return self._geometry

    @property
    def points(self) -> np.ndarray:

        return self._points

    @property
    def lines(self) -> np.ndarray:

        return self._lines

    @property
    def points_world(self) -> np.ndarray:
        
        return self._points_world

    def _update_transforms(self) -> None:
             
        super()._update_transforms()
        self._update_geometry()

    def _update_geometry(self) -> None:
        
        if self.fast_mode:
            self._points_world = (self.T_world.R @ self._points.T + self.T_world.r[:,None]).T
        else:
            self._geometry.points = o3d.utility.Vector3dVector((self.T_world.R @ self._points.T + self.T_world.r[:,None]).T)
            self._points_world = np.asarray(self._geometry.points)

    def save_as_trianglemesh(self, filename: str, write_ascii: bool = False, compressed: bool = False, print_progress: bool = False):

        # transform geometry vertices to local frame
        self._geometry.points = o3d.utility.Vector3dVector(self._points)
 
        o3d.io.write_line_set(filename, self._geometry, write_ascii, compressed, print_progress)

        # transform geometry vertices back to world frame
        self._update_geometry()

class TbCylinderCollidable(TbCollidable):

    def __init__(self, radius: float = 1, height: float = 1, height_subdivisions: int = 1, radial_subdivisions: int = 10, **kwargs) -> None:

        self._radius = radius
        self._height = height

        geometry = o3d.geometry.LineSet.create_from_triangle_mesh(o3d.geometry.TriangleMesh.create_cylinder(radius = radius, 
                                                                  height = height, 
                                                                  resolution = radial_subdivisions, 
                                                                  split = height_subdivisions, 
                                                                  create_uv_map = False))
        
        TbCollidable.__init__(self, geometry = geometry, **kwargs)

    @property
    def radius(self) -> float:

        return self._radius

    @property
    def height(self) -> float:

        return self._height
   

class TbSphereCollidable(TbCollidable):

    def __init__(self, radius: float = 1, subdivisions: int = 10, **kwargs) -> None:

        self._radius = radius

        geometry = o3d.geometry.LineSet.create_from_triangle_mesh(o3d.geometry.TriangleMesh.create_sphere(radius = radius,
                                                                  resolution = subdivisions, 
                                                                  create_uv_map = False))

        TbCollidable.__init__(self, geometry = geometry, **kwargs)

    @property
    def radius(self) -> float:

        return self._radius
        

class TbBoxCollidable(TbCollidable):

    def __init__(self, dimensions: list[float] = [1, 1, 1], **kwargs) -> None:

        self._dimensions = dimensions
        
        geometry = o3d.geometry.LineSet.create_from_triangle_mesh(o3d.geometry.TriangleMesh.create_box(width = dimensions[0],   # x-direction
                                                                  height = dimensions[1],  # y-direction
                                                                  depth = dimensions[2],   # z-direction
                                                                  create_uv_map = False))

        TbCollidable.__init__(self, geometry = geometry, **kwargs)

    @property
    def dimensions(self) -> list[float]:

        return self._dimensions