from __future__     import annotations
from alphashape     import alphashape as alphatriangulation
from .TbObject      import TbObject
from .TbPoint       import TbPoint
import open3d as o3d
import numpy  as np
from copy import deepcopy

class TbGeometry(TbObject):

    def __init__(self, geometry: o3d.geometry.TriangleMesh = None, mass: float = 0, com: TbPoint = None, **kwargs) -> None:

        if com is None:
            com = TbPoint()

        self._geometry       = geometry
        self._mass           = mass
        self._com            = com
        self._vertices       = deepcopy(np.asarray(self._geometry.vertices))
        self._triangles      = deepcopy(np.asarray(self._geometry.triangles))
        self._vertices_world = np.empty(self._vertices.shape)

        TbObject.__init__(self, children = [com], **kwargs)

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
        self._geometry = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self._vertices), o3d.utility.Vector3iVector(self._triangles))
        
    @property
    def geometry(self) -> o3d.geometry.TriangleMesh:

        return self._geometry

    @property
    def mass(self) -> float:

       return self._mass

    @mass.setter
    def mass(self, value: float):

        self._mass = value

    @property
    def com(self) -> TbPoint:

        return self._com

    @com.setter
    def com(self, value: TbPoint) -> None:

        self._com = value

    @property
    def vertices(self) -> np.ndarray:

        return self._vertices

    @property
    def faces(self) -> np.ndarray:

        return self._triangles

    @property
    def vertices_world(self) -> np.ndarray:
        
        return self._vertices_world

    def _update_transforms(self) -> None:
             
        super()._update_transforms()
        self._update_geometry()

    def _update_geometry(self) -> None:
        
        if self.fast_mode:
            self._vertices_world = (self.T_world.R @ self._vertices.T + self.T_world.r[:,None]).T
        else:
            self._geometry.vertices = o3d.utility.Vector3dVector((self.T_world.R @ self._vertices.T + self.T_world.r[:,None]).T)
            self._vertices_world = np.asarray(self._geometry.vertices)

    def save_as_trianglemesh(self, filename: str, write_ascii: bool = False, compressed: bool = False, write_vertex_normals: bool = True,
                              write_vertex_colors: bool = True, write_triangle_uvs: bool = True, print_progress: bool = False):

        # transform geometry vertices to local frame
        self._geometry.vertices = o3d.utility.Vector3dVector(self._vertices)

        # compute normals (required for stl)
        self._geometry.compute_triangle_normals()
 
        o3d.io.write_triangle_mesh(filename, self._geometry, write_ascii, compressed, write_vertex_normals, write_vertex_colors, write_triangle_uvs, print_progress)

        # transform geometry vertices back to world frame
        self._update_geometry()

class TbCylinder(TbGeometry):

    def __init__(self, radius: float = 1, height: float = 1, height_subdivisions: int = 1, radial_subdivisions: int = 10, **kwargs) -> None:

        self._radius = radius
        self._height = height

        geometry = o3d.geometry.TriangleMesh.create_cylinder(radius = radius, 
                                                             height = height, 
                                                             resolution = radial_subdivisions, 
                                                             split = height_subdivisions, 
                                                             create_uv_map = False)
        
        TbGeometry.__init__(self, geometry = geometry, **kwargs)

    @property
    def radius(self) -> float:

        return self._radius

    @property
    def height(self) -> float:

        return self._height
   

class TbSphere(TbGeometry):

    def __init__(self, radius: float = 1, subdivisions: int = 10, **kwargs) -> None:

        self._radius = radius

        geometry = o3d.geometry.TriangleMesh.create_sphere(radius = radius,
                                                           resolution = subdivisions, 
                                                           create_uv_map = False)

        TbGeometry.__init__(self, geometry = geometry, **kwargs)

    @property
    def radius(self) -> float:

        return self._radius
        

class TbBox(TbGeometry):

    def __init__(self, dimensions: list[float] = [1, 1, 1], **kwargs) -> None:

        self._dimensions = dimensions
        
        geometry = o3d.geometry.TriangleMesh.create_box(width = dimensions[0],   # x-direction
                                                        height = dimensions[1],  # y-direction
                                                        depth = dimensions[2],   # z-direction
                                                        create_uv_map = False)

        TbGeometry.__init__(self, geometry = geometry, **kwargs)

    @property
    def dimensions(self) -> list[float]:

        return self._dimensions


class TbTrianglemesh(TbGeometry):

    def __init__(self, filename: str = '', enable_post_processing: bool = False,  print_progress: bool = False, **kwargs) -> None:
        # supported formats are obj, ply, stl, off, gltf, glb, fbx
        
        self._filename = filename
        
        geometry = o3d.io.read_triangle_mesh(filename, enable_post_processing=enable_post_processing, print_progress=print_progress)
        geometry.scale(0.001, center=(0, 0, 0))
        geometry.remove_duplicated_vertices()
        geometry. remove_duplicated_triangles()
        
        TbGeometry.__init__(self, geometry=geometry, **kwargs)

    @property
    def filename(self) -> str:

        return self._filename


class TbAlphashape(TbGeometry):

    def __init__(self, points: list[list[float]] = [], alpha: float = 1, **kwargs) -> None:

        self._points = points
        
        triangulation = alphatriangulation(points, alpha)

        geometry = o3d.geometry.TriangleMesh(vertices = o3d.utility.Vector3dVector(triangulation.vertices),
                                             triangles = o3d.utility.Vector3iVector(triangulation.faces))

        TbGeometry.__init__(self, geometry=geometry, **kwargs)

    @property
    def points(self) -> list[list[float]]:

        return self._points


class TbTethergeometry(TbGeometry):

    def __init__(self, radius: float = 0.001, height_subdivisions: int = 1, radial_subdivisions: int = 10, **kwargs) -> None:

        self._radius = radius

        geometry = o3d.geometry.TriangleMesh.create_cylinder(radius = radius, 
                                                             height = 1, 
                                                             resolution = radial_subdivisions, 
                                                             split = height_subdivisions, 
                                                             create_uv_map = False)

        self._scale = np.ones(3)[:, None]    # scale factor for changing the height of the tether cylinder

        TbGeometry.__init__(self, geometry = geometry, **kwargs)

    @property
    def radius(self) -> float:

        return self._radius

    def _update_geometry(self) -> None:

        if hasattr(self._parent, 'length'):
            self._scale[2] = self._parent.length
            
        self._geometry.vertices = o3d.utility.Vector3dVector((self.T_world.R @ (self._vertices.T * self._scale) + self.T_world.r[:,np.newaxis]).T)