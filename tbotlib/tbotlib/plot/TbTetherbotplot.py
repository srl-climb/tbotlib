from __future__        import annotations
from ..tetherbot       import TbObject, TbPart, TbGeometry, TbPoint
from .Plot3d           import Plot3d
from .TbObjectplot     import TbObjectplot
from .TbFrameplot      import TbFrameplot
from .TbGeometryplot   import TbGeometryplot
from .TbPointplot      import TbPointplot
from .TbNameplot       import TbNameplot
from typing            import Type

class TbTetherbotplot(TbObjectplot):

    def __init__(self, data: Type[TbObject], **kwargs) -> None:
 
        self._objects = [data] + data.get_all_children(filter_duplicates=True)

        super().__init__(data, **kwargs)

    @property
    def data(self) -> Type[TbObject]:

        return self._data

    @data.setter
    def data(self, value: Type[TbObject]) -> None:

        self._data    = value
        self._objects = [value] + value.get_all_children(filter_duplicates=True)
        self.update()

    def _create(self, draw_frames: bool = True, draw_points: bool = True, draw_geometries: bool = True, draw_names: bool = False) -> None:
    
        self._plots: dict[Type[TbObject], list[Type[Plot3d]]] = {}

        for object in self._objects:

            self._plots[object] = []

            if isinstance(object, (TbPart, TbPoint)):
                if draw_points:
                    self._plots[object].append(TbPointplot(object))
                if draw_frames:
                    self._plots[object].append(TbFrameplot(object))
                if draw_names:
                    self._plots[object].append(TbNameplot(object))

            if isinstance(object, TbGeometry):
                if draw_geometries:
                    self._plots[object].append(TbGeometryplot(object))

    def update(self) -> None:
        
        for object in self._objects:
            for plot in self._plots[object]:
                plot.update()


    """ if __name__ == '__main__':

    from core import show, draw
    import matplotlib.pyplot as plt
    from ..tools import tic, toc

    tetherbot = TbTetherbot.example()
    plot = TbTetherbotplot(tetherbot)

    #show()

    show(block=False, animate=True)

    for i in range(1000):
        tic()
        draw()
        toc()
        tic()
        tetherbot.platform.T_local = tetherbot.platform.T_local.rotate(2,1,0)
        toc()
        tic()
        plot.update()
        toc()
        print() """

    """ gripper = TbGripper.example()
    plot = TbTetherbotplot(gripper, draw_names=True)

    #show()

    show(block=False, animate=True)

    for i in range(1000):

        draw()
        gripper.T_local = gripper.T_local.translate([0.01,0,0])
        plot.update()
        plt.pause(0.5) """

    """ parent  = TbPoint()
    point_1 = TbPoint(parent=parent, T_local=[0,0,0])
    point_2 = TbPoint(parent=parent, T_local=[1,0,0])
    tether  = TbTether(anchorpoints=[point_1, point_2], geometries=[TbTethergeometry(radius=0.005)])

    plot = TbTetherbotplot(parent)
    #plot = TbPointplot(point_1)
    show(block=False, animate=True)

    for i in range(100):

        draw()
        point_1.T_local = point_1.T_local.translate([0.01,0,0])
        point_2.T_local = point_2.T_local.translate([0,0.01,0])
        plot.update()
    """
    