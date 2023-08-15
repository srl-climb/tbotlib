from __future__             import annotations
from mpl_toolkits.mplot3d   import Axes3D
import matplotlib.pyplot    as plt

class Plot3d():

    def __init__(self, ax: Axes3D = None, xlabel: str = 'x', ylabel: str = 'y', zlabel: str = 'zlabel', **kwargs) -> None:
        
        if ax is None:
            
            # get current figure, or create one
            self._fig = plt.gcf()
            
            # find the first 3D axis
            self._ax = next((ax for ax in self._fig.axes if isinstance(ax, Axes3D)), None)
            
            # create a new 3D axis, if none was found
            if self._ax is None:
                self._ax = plt.axes(projection='3d')
                self._ax.set_box_aspect([1,1,1])
                self._ax.set_xlabel(xlabel)
                self._ax.set_ylabel(ylabel)
                self._ax.set_zlabel(zlabel)
        
        else:
            self._ax  = ax
            self._fig = ax.get_figure()

        self._create(**kwargs)
        self.update()

    def _create(self) -> None:

        pass

    def update(self) -> None:

        pass