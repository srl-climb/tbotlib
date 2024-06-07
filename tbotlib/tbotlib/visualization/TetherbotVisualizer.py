from __future__     import annotations
from ..tetherbot    import TbObject, TbTetherbot
from typing         import Type
import open3d as o3d
import numpy  as np
import msvcrt

class TetherbotVisualizer:

    def __init__(self, tbobject: Type[TbObject]) -> None:

        self._vi = o3d.visualization.Visualizer()
        self._vi.create_window()
        
        self._render = self._vi.get_render_option()
        self._render.background_color = np.array((0.1,0.1,0.1))
        self._render.light_on = True
        self._render.mesh_show_wireframe = True

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
                    self._vi.add_geometry(item.geometry)

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

    def save_camera_parameters(self, file: str):
        
        self._read_keyboard()
        print('Press Enter to save camera paramters.')

        parameters = None
        while self.opened:
            self.update()

            if self._read_keyboard() == b'\r':
                parameters = self._vi.get_view_control().convert_to_pinhole_camera_parameters()
                break

        if parameters is not None:
            o3d.io.write_pinhole_camera_parameters(file, parameters)
            print('Camera parameters saved under:', file)

    def load_camera_parameters(self, file: str) -> None:

        self._vi.get_view_control().convert_from_pinhole_camera_parameters(o3d.io.read_pinhole_camera_parameters(file))

    def set_background_color(self, value: list) -> None:

        self._render.background_color = value

    def capture_screen_image(self, file: str) -> None:

        self._vi.capture_screen_image(file , do_render=True)

    @staticmethod
    def _read_keyboard() -> str:

        if msvcrt.kbhit():
            return msvcrt.getch()
        else:
            return None

        




