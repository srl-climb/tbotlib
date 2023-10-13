from __future__     import annotations
from ..tetherbot    import TbObject, TbTetherbot
from typing         import Type
import open3d as o3d
import numpy  as np


class TetherbotVisualizer:

    def __init__(self, tbobject: Type[TbObject]) -> None:

        self._vi = o3d.visualization.Visualizer()
        self._vi.create_window()
        
        self._render = self._vi.get_render_option()
        self._render.background_color = np.array((0.1,0.1,0.1))
        self._render.light_on = True
        self._render.mesh_show_wireframe = True

	# causes error depending on open3d version!
        #self._material = o3d.visualization.rendering.MaterialRecord()
        #self._material.set_default_properties()

        self._tbobject = None
        self._geometries = []

        self._opened = True
        
        self.add_tbobject(tbobject)

    @property
    def opened(self) -> bool:

        return self._opened
        
    def __del__(self) -> None:

        self._vi.destroy_window()

    def add_tbobject(self, tbobject: Type[TbObject] = None) -> None:

        self.remove_tbobject()

        if tbobject is not None:
            self._tbobject = tbobject

            for item in [tbobject] + tbobject.get_all_children():
                if hasattr(item, 'geometry'):
                    self._geometries.append(item.geometry)

            for geometry in self._geometries:
                self._vi.add_geometry(geometry)

    def remove_tbobject(self) -> None:

        if self._tbobject is not None:
            for geometry in self._geometries:
                self._vi.remove_geometry(geometry)
            
            self._geometries = []

            self._tbobject   = None

    def update(self) -> None:

        for geometry in self._geometries:
            self._vi.update_geometry(geometry)
        
        self._opened = self._vi.poll_events()

        self._vi.update_renderer()
    
    def run(self) -> None:

        self._vi.run()

    def debug_move(self):
        
        if isinstance(self._tbobject, TbTetherbot):
            for _ in range(100):
                self._tbobject.platform.T_local = self._tbobject.platform.T_local.rotate(1,0,0)
                self.update()

                if not self._opened:
                    break

    def close(self) -> None:

        self._vi.close()



