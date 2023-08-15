from __future__           import annotations
from ..matrices           import TransformMatrix
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy             as np


class cframe():

    @property
    def T(self) -> TransformMatrix:
        
        return self._T

    @T.setter
    def T(self, value: TransformMatrix) -> None:
        
        self._T = value
        self._draw()

    @property
    def scale(self) -> float:
        
        return self.scale

    @scale.setter
    def scale(self, value: float) -> None:

        self._scale = value
        self._draw()

    def __init__(self, T: TransformMatrix, parent: plt.Axes = None, scale: float = 1) -> None:

        if parent is None:
            
            # get current figure, or create one
            fig = plt.gcf()

            # find the first 3D axis
            self.parent = next((ax for ax in fig.axes if isinstance(ax, Axes3D)), None)

            # create a new 3D axis, if none was found
            if self.parent is None:
                self.parent = plt.axes(projection='3d')
                self.parent.set_box_aspect([1,1,1])
                self.parent.set_xlabel('x')
                self.parent.set_ylabel('y')
                self.parent.set_zlabel('z')
        
        else:
            self.parent  = parent

        self._scale = scale
        self._E     = np.eye(3) 
        self._T     = T

        self._xline = self.parent.plot([0,1], [0,0], [0,0], 'r')[0]
        self._yline = self.parent.plot([0,0], [0,1], [0,0], 'g')[0]
        self._zline = self.parent.plot([0,0], [0,0], [0,1], 'b')[0]

        self._draw()

    def _draw(self) -> None:

        E = self._T.R @ self._E * self._scale
        
        self._xline.set_xdata([self._T.r[0], E[0, 0]+self._T.r[0]])
        self._yline.set_xdata([self._T.r[0], E[0, 1]+self._T.r[0]])
        self._zline.set_xdata([self._T.r[0], E[0, 2]+self._T.r[0]])

        self._xline.set_ydata([self._T.r[1], E[1, 0]+self._T.r[1]])
        self._yline.set_ydata([self._T.r[1], E[1, 1]+self._T.r[1]])
        self._zline.set_ydata([self._T.r[1], E[1, 2]+self._T.r[1]])

        self._xline.set_3d_properties([self._T.r[2], E[2, 0]+self._T.r[2]])
        self._yline.set_3d_properties([self._T.r[2], E[2, 1]+self._T.r[2]])
        self._zline.set_3d_properties([self._T.r[2], E[2, 2]+self._T.r[2]])

    def set_visible(self, value: bool =  True) -> None:

        self._xline.set_visible(value)
        self._yline.set_visible(value)
        self._zline.set_visible(value)

if __name__ == '__main__':

    T = TransformMatrix()

    c1 = cframe(T)
    T.rotate(45,10,0)
    T.translate([1,1,1])
    c2 = cframe(T, scale=0.5)
    plt.show()


