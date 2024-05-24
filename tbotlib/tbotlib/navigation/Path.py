from __future__ import annotations
from ..matrices import TransformMatrix
from ..tools    import cframe
import matplotlib.pyplot as plt
import numpy             as np

class Path():

    def __init__(self, coordinates: list[tuple]) -> None:

        self._coordinates = np.array(coordinates)

    @property
    def coordinates(self) -> np.ndarray:

        return self._coordinates

    @property
    def length(self) -> int:

        return len(self._coordinates)

    def append(self, coordinate: tuple) -> Path:

        self._coordinates = np.concatenate((self._coordinates, [coordinate]), axis=0)

        return self

    def replace(self, coordinate: tuple, idx: int = -1) -> Path:

        self._coordinates[idx] = coordinate

        return self


class Path6(Path):

    def __init__(self, coordinates: list[tuple]) -> None:

        super().__init__(coordinates)

        self._poses       = []
        for coordinate in self._coordinates:
            self._poses.append(TransformMatrix(coordinate))

    @property
    def coordinates(self) -> np.ndarray:

        return self._coordinates

    @property
    def poses(self) -> list[TransformMatrix]:

        return self._poses

    def append(self, coordinate: tuple) -> Path:

        super().append(coordinate)

        self._poses.append(TransformMatrix(coordinate))

        return self

    def replace(self, coordinate: tuple, idx: int = -1) -> Path:

        super().replace(coordinate, idx)

        self._poses[idx] = TransformMatrix(coordinate)

        return self

    def debug_plot(self, ax: plt.Axes = None):

        # Plot x, y, z
        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(projection='3d') 

        ax.set_box_aspect([1,1,1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.plot(self._coordinates[:,0], self._coordinates[:,1], self._coordinates[:,2], color = "black")

        for pose in self.poses:
            cframe(pose, parent = ax, scale = 0.5)
        
        plt.show()


class ClimbPath(Path):

    def __init__(self, stances: list[tuple]) -> None:
        
        super().__init__(stances)

    @property
    def stances(self) -> np.ndarray:

        return self._coordinates


