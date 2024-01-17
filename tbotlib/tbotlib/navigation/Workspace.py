from __future__  import annotations
from ..matrices  import NdTransformMatrix
from ..tetherbot import TbTetherbot
from ..plot      import TbTetherbotplot
from ..tools     import basefit, insideout
from copy        import deepcopy
from typing      import Tuple
import matplotlib.pyplot as plt
import numpy             as np

class Workspace():

    def __init__(self, bounds: np.ndarray, scale: np.ndarray, mode: str = None) -> None:

        self._bounds = np.array(bounds)
        self._scale  = np.array(scale)
        self._ndim   = len(scale)
        self._grid   = None
        self._vals   = None
        self._mode   = mode

    @property
    def mode(self) -> str:

        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:

        self._mode = value

    def calculate(self, T: NdTransformMatrix) -> Tuple[float, np.ndarray]:

        # generate grid coordinates
        self._generate_grid(T)

        if self._mode == 'max' or self._mode is None:

            # evaluate each coordinate of the grid
            for idx in range(len(self._grid)):
                self._eval(idx)

            max_idx   = np.argmax(self._vals)

            return self._vals[max_idx], self._grid[max_idx]

        if self._mode == 'first':

            # evaluate until first positive
            for idx in range(len(self._grid)):
                val, coordinate = self._eval(idx)
                
                if val >= 0:
                    return val, coordinate

            return -1, None  
        
        pass

    def _generate_grid(self, T: NdTransformMatrix) -> None:

        coordinates = []
        
        # for each dimension of the grid
        for scale, bound in zip(self._scale, self._bounds):

            # calculate number of coordinates (+1 for the endpoint)
            num = int((bound[1]-bound[0])/scale) + 1

            # calculate coordinates in n-th direction
            coordinates.append(insideout(np.linspace(bound[0], bound[1], num, endpoint=True)))
            # Note: Insideout sorts the coordinates, starting from the center. This will lead to
            #       improved performance, when looking for the first stable pose in the workspace
            #       since a stable pose is much more likely to be found in the center instead of
            #       the edge of the grid.
            #       coordinates.append(np.linspace(bound[0], bound[1], num, endpoint=True))

        # generate grid
        coordinates.reverse()
        self._grid = np.flip(np.array(np.meshgrid(*coordinates, indexing='xy')).T.reshape(-1, self._ndim), axis=1)
        # Note: Reversing and flipping will sort the grid coordinates in a way, that all rotations
        #       for a given position will be evaluated first before moving to the next position. This
        #       will improve performance, when looking for the first stable pose in the workspace
        #       self._grid = np.array(np.meshgrid(*coordinates)).T.reshape(-1, self._ndim)

        # generate value array
        self._vals = np.zeros(len(self._grid))-1

        # transform to world coordinates
        self._grid = T.forward_transform(self._grid, axis = 1, copy = False)


    def _eval(self, idx: int) -> Tuple[float, np.ndarray]:

        self._vals[idx] = 1

        return self._vals[idx], self._grid[idx]

    def debug_print(self) -> None:

        print()
        print(self)
        print('shape:     ', self._grid.shape)
        print('scale:     ', self._scale)
        print('bounds:')
        print(self._bounds)


class TbWorkspace(Workspace):

    def __init__(self, scale: np.ndarray = np.ones(6), padding: np.ndarray = np.zeros(6), mode_2d: bool = False,  **kwargs) -> None:
        
        self._padding = np.array(padding)
        self._mode_2d = mode_2d

        super().__init__(bounds = np.empty((6,2)), scale = scale, **kwargs)

    def calculate(self, tetherbot: TbTetherbot) -> Tuple[float, np.ndarray]:
        
        self._tetherbot = deepcopy(tetherbot)

        # Transformation of the workspace grid
        R = np.eye(6)
        r = np.zeros(6)
        r[:3], R[:3,:3] = basefit(self._tetherbot.A_world, axis=0)
        r[3:]           = self._tetherbot.platform.T_world.decompose()[3:]

        T = NdTransformMatrix(r, R)

        # Bounds of the workspace grid
        # Position bounds
        A_grid = T.inverse_transform(np.pad(self._tetherbot.A_world ,((0,3),(0,0))), axis=0)[:3,:]
        
        median   = (np.max(A_grid, axis=1) + np.min(A_grid, axis=1))*0.5
        variance = np.clip((np.max(A_grid, axis=1) - np.min(A_grid, axis=1))*0.5 + self._padding[None,:3], a_min=0, a_max=np.inf)
        
        self._bounds[:3,0] = median - variance
        self._bounds[:3,1] = median + variance

        # Rotation bounds
        median   = 0 #self._tetherbot.platform.T_world.decompose()[3:]
        variance = np.clip(np.ones(3) * 180 + self._padding[None,3:], a_min=0, a_max=180)

        self._bounds[3:,0] = median - variance
        self._bounds[3:,1] = median + variance

        if self._mode_2d:
            T._T[2,-1] = self._tetherbot.platform.T_world.r[2]
            self._bounds[2,0] = 0
            self._bounds[2,1] = 0
            self._bounds[3,0] = 0 #self._tetherbot.platform.T_world.decompose()[3]
            self._bounds[3,1] = 0 #self._tetherbot.platform.T_world.decompose()[3]
            self._bounds[4,0] = 0 #self._tetherbot.platform.T_world.decompose()[4]
            self._bounds[4,1] = 0 #self._tetherbot.platform.T_world.decompose()[4]

        return super().calculate(T)

    def _eval(self, idx: int) -> Tuple[float, np.ndarray]:

        self._tetherbot.platform.T_world = self._tetherbot.platform.T_world.compose(self._grid[idx])

        self._vals[idx] = self._tetherbot.stability()[0]

        return self._vals[idx], self._grid[idx]

    def debug_plot(self) -> None:

        # Plot x, y, z
        fig = plt.figure()
        ax  = fig.add_subplot(projection='3d') 
        xyz = np.unique(self._grid[:,:3], axis = 0)

        ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Tetherbot
        # TbTetherbotplot(self._tetherbot, ax=ax)

        # Plot theta_x, y, z
        fig = plt.figure()
        ax  = fig.add_subplot(projection='3d') 
        xyz = np.unique(self._grid[:,3:], axis = 0)

        ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
        ax.set_xlabel('theta_x')
        ax.set_ylabel('theta_y')
        ax.set_zlabel('theta_z')

        plt.show()