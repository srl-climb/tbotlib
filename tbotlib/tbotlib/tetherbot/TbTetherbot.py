from __future__      import annotations
from ..fdsolvers     import QuadraticProgram, HyperPlaneShifting, CornerCheck, QuickHull, AdaptiveCWSolver
from ..tools         import Ring, Mapping, hyperRectangle, basefit, inpie, inrectangle, perp, ang3
from ..matrices      import StructureMatrix, TransformMatrix, rotM
from ..visualization import TetherbotVisualizer
from .TbObject       import TbObject
from .TbPlatform     import TbPlatform
from .TbGripper      import TbGripper
from .TbTether       import TbTether
from .TbWall         import TbWall
from .TbPart         import TbPart
from .TbGeometry     import TbCylinder, TbSphere, TbTethergeometry
from typing          import Tuple
from scipy.optimize  import least_squares
import numpy as np

class TbTetherbot(TbObject):

    def __init__(self, platform: TbPlatform = None, grippers: list[TbGripper] = None, tethers: list[TbTether] = None, wall: TbWall = None,
                 w: np.ndarray = None, W: np.ndarray = None, mapping: Mapping = None, aorder: Ring = None, mode_2d: bool = True, l_min: float = 0.012, l_max: float = 2, **kwargs) -> None:
        
        super().__init__(children = [platform, wall], **kwargs)
        # do not pass tethers as children, as they will become children of the anchorpoints later

        self._platform  = platform
        self._wall      = wall
        self._tethers   = tethers
        self._grippers  = grippers
        self._m         = len(tethers)
        self._k         = len(grippers)
        self._n         = 6
        self._A         = np.empty((3, self._m))
        self._B         = np.empty((3, self._m))
        self._C         = np.empty((3, wall.n))
        self._f_max     = np.empty(self._m)
        self._f_min     = np.empty(self._m)
        self._tensioned = np.full(self._m, True, dtype=bool)
        self._AT        = StructureMatrix()
        self._fdsolver  = QuadraticProgram(self._m, self._n)
        self._cwsolver  = AdaptiveCWSolver(self._m, self._n)
        self._aorder    = aorder
        self._mode_2d   = mode_2d
        self._l_min     = l_min
        self._l_max     = l_max
        
        if w is None:
            self._w = np.zeros(self._n)
        else:
            self._w = np.array(w)

        if W is None:
            self.W = np.zeros((1,self._n))
        else:
            self.W = np.array(W)

        if mapping is None:
            self._mapping = Mapping([[i//2,i] for i in range(self._m)])
        elif isinstance(mapping, Mapping):
            self._mapping = mapping
        else:
            self._mapping = Mapping(mapping)
        
        for i in range(self._m):
            tethers[i].anchorpoints = [self.grippers[self._mapping.a_to_b[i,0]].anchorpoint, 
                                       self.platform.anchorpoints[self._mapping.a_to_b[i,1]]]

        self.place_all(range(self.k))

        self._update_transforms()
        

    @property
    def platform(self) -> TbPlatform:
        
        return self._platform

    @property
    def grippers(self) -> list[TbGripper]:

        return self._grippers

    @property
    def tethers(self) -> list[TbTether]:

        return self._tethers

    @property
    def wall(self) -> TbWall:

        return self._wall

    @property   # Number of tethers
    def m(self) -> int:

        return self._m

    @property   # Degrees of freedom
    def n(self) -> int:

        return self._n

    @property   # Number of grippers
    def k(self) -> int:

        return self._k

    @property   # Positions of the gripper tether anchorpoints
    def A_world(self) -> np.ndarray:

        for i in range(self._m):
            self._A[:, i] = self.grippers[self._mapping.a_to_b[i,0]].anchorpoint.r_world

        return self._A

    @property   # Positions of the platform tether anchorpoints
    def B_local(self) -> np.ndarray:

        for i in range(self._m):
            self._B[:,i] = self.platform.anchorpoints[self._mapping.a_to_b[i,1]].r_local

        return self._B
    
    @property   # Positions of the platform tether anchorpoints
    def B_world(self) -> np.ndarray:

        for i in range(self._m):
            self._B[:,i] = self.platform.anchorpoints[self._mapping.a_to_b[i,1]].r_world

        return self._B

    @property   # Positions of all holds
    def C_world(self) -> np.ndarray:

        for i in range(self.wall.n):
            self._C[:,i] = self.wall.holds[i].r_world

        return self._C

    @property   # Tether vectors
    def L(self) -> np.ndarray:

        return self.A_world - self.B_world

    @property   # Tether unit vectors
    def N(self) -> np.ndarray:

        return self.L / np.linalg.norm(self.L, axis=0)

    @property   # Tether lengths
    def l(self) -> np.ndarray:

        return np.linalg.norm(self.L, axis=0)

    @property   # Structure matrix
    def AT(self) -> np.ndarray:

        return self._AT(self.L, self.B_world)

    @property   # Tether min forces
    def f_min(self) -> np.ndarray:

        for i in range(self._m):
            self._f_min[i] = self._tethers[i].f_min

        return self._f_min

    @property   # Tether max forces
    def f_max(self) -> np.ndarray:

        for i in range(self._m):
            self._f_max[i] = self._tethers[i].f_max
        
        return self._f_max

    @property   # wrench
    def w(self) -> np.ndarray:

        return self._w

    @w.setter  
    def w(self, w: np.ndarray) -> None:

        self._w = np.array(w)

    @property
    def tensioned(self) -> np.ndarray:

        return self._tensioned

    @property
    def mapping(self) -> Mapping:

        return self._mapping

    @property
    def aorder(self) -> Ring:

        return self._aorder
    
    @property
    def l_min(self) -> float:

        return self._l_min
    
    @property
    def l_max(self) -> float:

        return self._l_max

    def forces(self, w: np.ndarray = None) -> tuple[np.ndarray, int]:

        if w is None:
            w = self._w
        
        return self._fdsolver.eval(self.AT, w, self.f_min, self.f_max)

    def gripperforces(self, w: np.ndarray = None) -> tuple[np.ndarray, int]:

        f, exitflag = self.forces(w)

        # tether force vectors of each gripper
        F = f * self.N
        
        # tether forces vectors sorted by gripper
        F = np.swapaxes(F[:, list(self._mapping.from_a.values())], 0, 1)
        
        return np.linalg.norm(np.sum(F, axis=2), axis=1), exitflag

    def stability(self, W: np.ndarray = None) -> Tuple(bool, float):

        if W is None:
            W = self.W

        l = self.l
        if all(self._l_min < l) and all(l < self._l_max):
            return self._cwsolver.eval(self.AT, W, self.f_min, self.f_max, self._tensioned)
        else:
            return False, -np.inf

    def tension(self, idx_gripper: int, value: bool) -> None:
        
        for i in self._mapping.from_a[idx_gripper]:
            self.tethers[i].tensioned = value

        for i in range(self._m):
            self._tensioned[i] = self._tethers[i].tensioned

    def pick(self, grip_idx: int, correct_pose: bool = False) -> None:
      
        #self.tension(grip_idx, False)   

        if correct_pose:    
            self.grippers[grip_idx].T_local = TransformMatrix(self.platform.arm.links[-1].T_local.T @  self.grippers[grip_idx].dockpoint.T_local.Tinv)
        else:
            # reset T_local to maintain T_world
            self.grippers[grip_idx].T_local = TransformMatrix(self.platform.arm.links[-1].T_world.Tinv @ self.grippers[grip_idx].T_world.T)

        self.grippers[grip_idx].parent  = self.platform.arm.links[-1]


    def place(self, grip_idx: int, hold_idx: int, correct_pose: bool = False) -> None:

        #self.tension(grip_idx, True)

        if correct_pose:
            self.grippers[grip_idx].T_local = TransformMatrix(self.wall.holds[hold_idx].grippoint.T_local.T @  self.grippers[grip_idx].grippoint.T_local.Tinv)
        else:
            # reset T_local to maintain T_world
            self.grippers[grip_idx].T_local = TransformMatrix(self.wall.holds[hold_idx].T_world.Tinv @ self.grippers[grip_idx].T_world.T)

        self.grippers[grip_idx].parent = self.wall.holds[hold_idx]

    def place_all(self, hold_idc: list[int], correct_pose: bool = True) -> None:

        for grip_idx, hold_idx in zip(range(self._k), hold_idc):
            self.place(grip_idx, hold_idx, correct_pose)

    def filter_holds(self, grip_idx: int, points: np.ndarray = None) -> np.ndarray:
        
        if points is None:
            points = self.C_world

        # base of the plane to filter the holds
        base = basefit(self.A_world, axis=0, output_format=1)
        # ez is perpendicular to plane

        # index of the gripper within the order
        order_idx = self._aorder.index(grip_idx)

        filter = np.full(points.shape[1], False)
        radius = self.platform.arm.workspace_radius

        # project all points and gripper positions on to the plane
        points  = base.inverse_transform(points, axis=0, copy=True).T[:,:2] 
        right_2 = base.inverse_transform(self.grippers[self._aorder[order_idx-2]].r_world)[:2]
        right_1 = base.inverse_transform(self.grippers[self._aorder[order_idx-1]].r_world)[:2]
        left_1  = base.inverse_transform(self.grippers[self._aorder[order_idx+1]].r_world)[:2]
        left_2  = base.inverse_transform(self.grippers[self._aorder[order_idx+2]].r_world)[:2]
        # Note: gripper positions should be anticlockwise

        # check middle
        filter = filter | inrectangle(right_1, right_1+perp(left_1-right_1)*radius, left_1+perp(left_1-right_1)*radius, left_1, points)
        
        # check to the right  
        right_angle = ang3(left_1-right_1, right_2-right_1)
        if right_angle < 90: # acute angle -> add points    
            filter = filter | inpie(right_1, radius, right_1-right_2, perp(left_1-right_1),  points)
        elif right_angle > 90: # obstuse angle -> remove points
            filter = filter & np.logical_not(inpie(right_1, np.inf*radius, perp(left_1-right_1), right_1-right_2,  points, mode='ex'))

        # check to the left
        left_angle = ang3(right_1-left_1, left_2-left_1)
        if left_angle < 90: # acute angle -> add points
            filter = filter | inpie(left_1, radius, perp(left_1-right_1), left_1-left_2, points) 
        elif left_angle > 90:
            filter = filter & np.logical_not(inpie(left_1, np.inf*radius, left_1-left_2, perp(left_1-right_1), points, mode='ex')) 

        return filter.nonzero()[0]
        
    def remove_all_geometries(self) -> None:

        for child in self.get_all_children():
            if isinstance(child, TbPart):
                child.remove_geometries()
        
    def toggle_fast_mode(self, value: bool) -> None:

        for child in self.get_all_children():
            child.fast_mode = bool(value)

    def get_gripper(self, id: str) -> TbGripper:

        if id.isnumeric():
            i = int(id)
        else:
            i = [gripper.name for gripper in self._grippers].index(id)

        return self._grippers[i]
    
    def get_tether(self, id: str) -> TbTether:

        if id.isnumeric():
            i = int(id)
        else:
            i = [tether.name for tether in self._tethers].index(id)

        return self._tethers[i]

    def debug_print(self) -> None:
        
        print()
        print(self.__class__, 'Debug Print')
        print('===========================')
        print('Number of tethers m:', self.m)
        print('Number of grippers k:', self.k)
        print('Degrees of freedom n:', self.n)
        print('Gripper anchor points A_world: ', np.round(self.A_world, 2))
        print('Platform anchor points B_world: ', np.round(self.B_world, 2))
        print('Platform pose:', np.round(self.platform.T_world.decompose(), 2))
        print('Max tether force:', self.f_max)
        print('Min tehter force:', self.f_min)
        print('Current tether force:', np.round(self.forces()[0], 2))
        print('Tensioned tethers:', self.tensioned)
        print('Mapping of a to b:', self.mapping.a_to_b)
        print('Order of the grippers:', self.aorder)
        print('Number of holds:', self.wall.n)
        print('Hold positions: ', np.round(self.C_world, 2))
        print('CW-solver:', self._cwsolver)
        print('FD-solver:', self._fdsolver)
        print('===========================')

    def debug_plot(self) -> None:

        vi = TetherbotVisualizer(self)
        vi.run()

    def fwk(self, l: np.ndarray, T0: TransformMatrix) -> TransformMatrix:

        x0 = T0.decompose()

        if self._mode_2d:
            # z, theta_x, theta_y are constant
            x = least_squares(self.fwk_fun2, 
                              x0=x0[[0,1,5]], 
                              args=(l, self.A_world, self.B_local, x0[2], x0[3], x0[4]), 
                              method='lm').x
            x = np.array([x[0], x[1], x0[2], x0[3], x0[4], x[2]])
        else:
            x = least_squares(self.fwk_fun, 
                              x0=x0, 
                              args=(l, self.A_world, self.B_local, self.platform.T_world.z),
                              method='lm').x
            
        self.platform.T_world = TransformMatrix(x)
        
        return self.platform.T_world
    
    def ivk(self, T: TransformMatrix) -> np.ndarray:

        return np.linalg.norm(self.A_world-T.r[:,np.newaxis]-T.R@self.B_local, axis=0)

    @staticmethod
    def fwk_fun(x: np.ndarray, l: np.ndarray, A_world: np.ndarray, B_local: np.ndarray, z) -> float:
        
        r = x[:3]
        R = rotM(x[3], x[4], x[5])

        # numeric objective function according to Pott 2018
        return np.square(np.linalg.norm( A_world - r[:,np.newaxis] - R @ B_local, axis=0)) - np.square(l)
        #return np.sum( np.square(np.linalg.norm( A_world - r[:,np.newaxis] - R @ B_local, axis=0)) - np.square(l) )**2

    @staticmethod
    def fwk_fun2(x: np.ndarray, l: np.ndarray, A_world: np.ndarray, B_local: np.ndarray, z: float, theta_x: float, theta_y: float) -> float:

        r = np.array([x[0], x[1], z])
        R = rotM(theta_x, theta_y, x[2])
        
        # numeric objective function according to Pott 2018
        return np.square(np.linalg.norm( A_world - r[:,np.newaxis] - R @ B_local, axis=0)) - np.square(l)

    @staticmethod
    def example() -> TbTetherbot:
        
        m       = 10
        k       =  5
        W       = hyperRectangle(np.array([5,5,5,0.5,0.5,0.5]), np.array([-5,-5,-5,-0.5,-0.5,-0.5]))
        mapping = [[0,0],[0,1],[1,2],[1,3],[3,4],[3,5],[4,6],[4,7],[2,8],[2,9]]
        aorder  = Ring([0,1,3,4,2]) #indices of the grippers counter clockwise

        tethers  = [TbTether.example() for _ in range(m)]
        grippers = [TbGripper.example() for _ in range(k)]
        platform = TbPlatform.example()
        wall     = TbWall.example()

        for gripper in grippers:
            gripper.add_geometry(TbCylinder(T_local = [0,0,0.015], radius = 0.015, height = 0.03))
            gripper.add_geometry(TbSphere(T_local = [0,0,0.03], radius=0.02))
        
        for tether in tethers:
            tether.add_geometry(TbTethergeometry(radius = 0.008))

        return TbTetherbot(platform=platform, grippers=grippers, tethers=tethers, wall=wall, W=W, mapping=mapping, aorder=aorder)


""" # place grippers onto wall
        for gripper, hold in zip(self._grippers, self._wall.holds):
            gripper.parent  = hold
            gripper.T_local = gripper.T_local.set_r(gripper.grippoint.r_local) """