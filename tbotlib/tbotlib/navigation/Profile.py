from __future__     import annotations
from ..matrices     import TransformMatrix
from ..tools        import interleave, slice, cframe, polysplinefit
from ..tetherbot    import TbTetherbot
from .Path          import Path
from .Smoother      import *
from typing         import Tuple
from math           import sqrt, ceil
from abc            import ABC, abstractmethod
from copy           import deepcopy
import numpy             as np
import matplotlib.pyplot as plt


class AbstractProfile(ABC):

    @abstractmethod
    def __init__(self, a_t: float, d_t: float, v_t: float, dt: float) -> None:
        '''
        a_t: target acceleration
        d_t: target deceleration
        v_t: target velocity
        dt:  time step
        '''
        self.a_t = a_t
        self.d_t = d_t
        self.v_t = v_t
        self.dt  = dt

    @abstractmethod
    def _calculate(self, coordinates: np.ndarray, **_) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        return 0, 0, 0, 0

    def calculate(self, path: Path = None, **kwargs) -> Profile:
        
        if path is not None:
            # Note: Due to historic reasons path stores coordinates row-wise while the Profile classes store them column wise
            return Profile(*self._calculate(path.coordinates.T, **kwargs))

        return None

class FastProfile(AbstractProfile):

    def __init__(self):

        super().__init__(0, 0, 0, 0)

    def _calculate(self, coordinates: np.ndarray, **_):

        c = coordinates[:, [0,-1]]
        v = np.zeros(c.shape)
        a = np.zeros(c.shape)
        t = np.zeros(2)
        
        return t, a, v, c

class Profile1(AbstractProfile):

    def __init__(self, a_t: float, d_t: float, v_t: float, dt: float) -> None:
        
        super().__init__(a_t, d_t, v_t, dt)

    def _calculate(self, coordinates: np.ndarray, **_) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # target distance
        q_t = coordinates[1] - coordinates[0]

        # distance/time to accelerate to v_t
        t_a = self.v_t/self.a_t
        q_a = 0.5*self.a_t*t_a**2

        # distance/time to decelerate from v_t
        t_d = self.v_t/self.d_t
        q_d = 0.5*self.d_t*t_d**2

        # distance long enough to accelerate to v_t?
        if q_d+q_a < q_t:

            # time to reach target distance
            t_t = t_a + t_d + (q_t-q_a-q_d)/self.v_t

        else:
            # find acceleration to deceleration transition where:
            # a_t*t_a = d_t*t_d
            # s_t = 0.5*a_t*t_a^2 + 0.5*d_t*t_d^2
            
            # new distance/time to accelerate and decelerate
            t_d = sqrt(q_t/(0.5*(self.d_t**2/self.a_t+self.d_t)))
            t_a = self.d_t/self.a_t*t_d

            # time to reach target distance
            t_t = t_a + t_d

        # number of time samples
        n = ceil(t_t/self.dt)+1

        # preallocated acceleration, position, ... arrays
        a = np.zeros(n)
        q = np.zeros(n)
        v = np.zeros(n)
        t = np.linspace(0, t_t, num=n, endpoint=True)

        # phase 0: acceleration
        i_0 = ceil(t_a/self.dt)
        a[:i_0] = self.a_t
        v[:i_0] = self.a_t*t[:i_0]
        q[:i_0] = 0.5*self.a_t*t[:i_0]**2

        # pase 1: maintain velocity
        i_1 = ceil((t_t-t_d)/self.dt)
        a[i_0:i_1] = 0
        v[i_0:i_1] = self.a_t*t_a
        q[i_0:i_1] = 0.5*self.a_t*t_a**2 + self.a_t*t_a*(t[i_0:i_1]-t_a)

        # phase 2: deceleration
        i_2 = ceil(t_t/self.dt)
        a[i_1:i_2] = -self.d_t
        v[i_1:i_2] = self.a_t*t_a - self.d_t*(t[i_1:i_2]-t_t+t_d)
        q[i_1:i_2] = 0.5*self.a_t*t_a**2 + self.a_t*t_a*(t_t-t_a-t_d) + self.a_t*t_a*(t[i_1:i_2]-t_t+t_d) - 0.5*self.d_t*(t[i_1:i_2]-t_t+t_d)**2

        # add last point       
        a[-1] = 0
        v[-1] = 0
        q[-1] = q_t 

        return t, a, v, q


class Profile3(AbstractProfile):

    def __init__(self, a_t: float, d_t: float, v_t: float, dt: float, smoother: AbstractSmoother = None) -> None:
        
        super().__init__(a_t, d_t, v_t, dt)

        if smoother is None:
            self.smoother = NotSmoother()
        else:
            self.smoother = smoother

    def _calculate(self, coordinates: np.ndarray, **_) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        coordinates = self.smoother.smooth(coordinates)

        if coordinates.shape[1] == 1:            
            return [0], [0], [0], np.array(coordinates, copy=True)

        # absolute and cumultative distance between the coordinates of the trajectory
        dist_abs = np.linalg.norm(np.diff(coordinates), axis=0)   
        dist_cum = np.cumsum(dist_abs)

        # create a profile over the cumultative distance
        t, a, v, q = Profile1(self.a_t, self.d_t, self.v_t, self.dt)._calculate([0, dist_cum[-1]])

        # map the positions of the profile to the trajectory
        i = np.searchsorted(dist_cum, q, sorter=None)
        # Note: Due to numerical errors, the last waypoint of s might be outside the cumultative distance.
        #       This results in an index i outside the length of coordinates, rounding prevents that

        q = coordinates[:,i+1] + ((coordinates[:,i]-coordinates[:,i+1])/dist_abs[i]) * (dist_cum[i]-q)
        # Note: The elements in dist are shifted by +1

        return t, a, v, q


class SlicedProfile6(AbstractProfile):

    def __init__(self, a_t: float, d_t: float, v_t: float, dt: float, smoother: AbstractSmoother) -> None:
        
        super().__init__(a_t, d_t, v_t, dt)

        self.smoother = smoother

    def _calculate(self, coordinates: np.ndarray, **_) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        # round to eliminate numerical erros which can interfere with correct slicing
        coordinates = np.round(coordinates, 4)

        # seperate the position and orientation from the pose
        xyz = coordinates[:3,:]
        abg = coordinates[3:,:]
        
        # slice the position/orientation arrays
        xyz_c, xyz_b = slice(xyz)
        abg_c, abg_b = slice(abg)

        # calculate the profiles for the position arrays
        if not xyz_b:
            
            xyz_a = [np.zeros((1))]
            xyz_v = [np.zeros((1))]
            xyz_q = [np.atleast_2d(np.ones(3)*xyz[:,0]).T]
        
        else:
        
            xyz_a = []
            xyz_v = []
            xyz_q = []

            for b in xyz_b:

                b = self.smoother.smooth(b)

                h = Profile3(self.a_t[0], self.d_t[0], self.v_t[0], self.dt)._calculate(b)
                xyz_a.append(h[1])
                xyz_v.append(h[2])
                xyz_q.append(h[3])

        # calculate the profiles for the orientation arrays
        if not abg_b:
            
            abg_a = [np.zeros((1))]
            abg_v = [np.zeros((1))]
            abg_q = [np.atleast_2d(np.ones(3)*abg[:,0]).T]
        
        else:

            abg_a = []
            abg_v = []
            abg_q = []

            for b in abg_b:

                b = self.smoother.smooth(b)

                h = Profile3(self.a_t[1], self.d_t[1], self.v_t[1], self.dt)._calculate(b)

                abg_a.append(h[1])
                abg_v.append(h[2])
                abg_q.append(h[3])

        # interleave the results   
        a = interleave(xyz_a, abg_a, xyz_c[0,0] > abg_c[0,0], stagger=1, fill=0)
        v = interleave(xyz_v, abg_v, xyz_c[0,0] > abg_c[0,0], stagger=1, fill=1)
        q = interleave(xyz_q, abg_q, xyz_c[0,0] > abg_c[0,0], stagger=1, fill=1)

        # create time stamps
        t = np.atleast_1d(np.arange(0, a.shape[1]) * self.dt)

        return t, a, v, q


class ParallelProfile6(AbstractProfile):

    def __init__(self, a_t: float, d_t: float, v_t: float, dt: float, smoother: AbstractSmoother) -> None:
        
        super().__init__(a_t, d_t, v_t, dt)

        self.smoother = smoother

    def _calculate(self, coordinates: np.ndarray, **_) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # seperate the position and orientation from the pose
        xyz = coordinates[:3,:]
        abg = coordinates[3:,:]

        # smooth the points
        xyz = self.smoother.smooth(xyz)
        abg = self.smoother.smooth(abg)

        # calculate the profiles
        xyz_t, xyz_a, xyz_v, xyz_q = Profile3(self.a_t[0], self.d_t[0], self.v_t[0], self.dt)._calculate(xyz)
        abg_t, abg_a, abg_v, abg_q = Profile3(self.a_t[1], self.d_t[1], self.v_t[1], self.dt)._calculate(abg)

        # combine the profiles
        xyz_n = len(xyz_t)    
        abg_n = len(abg_t) 

        if xyz_n > abg_n:
            n = xyz_n
            t = xyz_t
        else:
            n = abg_n
            t = abg_t

        a = np.zeros(2,n) 
        v = np.zeros(2,n)
        q = np.zeros(6,n)

        a[0,:xyz_n]  = xyz_a
        a[1,:abg_n]  = abg_a
        v[0,:xyz_v]  = xyz_v
        v[1,:abg_v]  = abg_v
        q[:3,xyz_n:] = xyz_q[:,-1]
        q[3:,xyz_n:] = abg_q[:,-1]
        q[:3,:xyz_n] = xyz_q
        q[3:,:abg_n] = abg_q

        return t, a, v, q


class ProfileQPlatform(AbstractProfile):

    def __init__(self, smoother: AbstractSmoother, dt: float, qlim: list = None, iter_max: int = 1) -> None:
        
        super().__init__(None, None, None, dt)

        self.smoother = smoother
        self.iter_max = iter_max
        self.qlim = np.array(qlim)

    def _calculate(self, coordinates: np.ndarray, tetherbot: TbTetherbot = None, **_) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if coordinates.shape[1] == 1:
            return np.array([0]), None, None, coordinates
          
        coordinates = self.smoother.smooth(coordinates) # column-wise

        # create independent copy
        tetherbot = deepcopy(tetherbot)

        # transform coordinates to joint space
        qs = self.to_jointspace(coordinates, tetherbot)

        spline_t, spline_qs, spline_qsd1, spline_qsd2, exitflag = polysplinefit(
            y = qs,
            step = self.dt,
            spline_dx = self.dt,
            spline_limits = self.qlim,
            spline_degree = 3,
            iter_max = self.iter_max,
            mode = 'segmental'
        )

        if exitflag == 0:
            print('Warning, profiler reached max iteration count before valid solution was found')
            print('duration in sec: ', spline_t[-1])
            print('spline q max: ', np.round(np.max(np.abs(spline_qs), axis=1),4))
            print('spline qd1 max: ', np.round(np.max(np.abs(spline_qsd1), axis=1),4))
            print('spline qd2 max: ', np.round(np.max(np.abs(spline_qsd2), axis=1),4))
        
        # debug code
        """ print(exitflag)
        print('target time in sec: ' ,t_t)
        print('spline dt in sec: ', self.dt)
        print('end time in sec: ', spline_t[-1])
        print('spline qsmax0: ', np.max(np.abs(spline_qs[0,:])))
        print('spline qsmax1: ', np.max(np.abs(spline_qs[1,:])))
        print('spline qsmax2: ', np.max(np.abs(spline_qs[2,:])))
        print('spline d1max0: ', np.max(np.abs(d1[0,:])))
        print('spline d1max1: ', np.max(np.abs(d1[1,:])))
        print('spline d1max2: ', np.max(np.abs(d1[2,:])))
        print('spline d2max0: ', np.max(np.abs(d2[0,:])))
        print('spline d2max1: ', np.max(np.abs(d2[1,:])))
        print('spline d2max2: ', np.max(np.abs(d2[2,:])))
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, spline_t[-1], qs.shape[1]), qs[0,:])
        ax.plot(np.linspace(0, spline_t[-1], qs.shape[1]), qs[1,:])
        ax.plot(np.linspace(0, spline_t[-1], qs.shape[1]), qs[2,:])
        fig, ax = plt.subplots()
        ax.plot(spline_t, spline_qs[0,:])
        ax.plot(spline_t, spline_qs[1,:])
        ax.plot(spline_t, spline_qs[2,:])
        fig, ax = plt.subplots()
        ax.plot(spline_t, d1[0,:])
        ax.plot(spline_t, d1[1,:])
        ax.plot(spline_t, d1[2,:])
        fig, ax = plt.subplots()
        ax.plot(spline_t, d2[0,:])
        ax.plot(spline_t, d2[1,:])
        ax.plot(spline_t, d2[2,:])
        print(np.round(spline_qs[1,0:10],4))
        print(np.round(d1[1,0:10],4))
        print(np.round(spline_t[0:10],4)) 
        plt.show()
        plt.close() """

        # transform jointspace to coordinates
        c = self.to_coordinatespace(spline_qs, tetherbot)

        t = spline_t
        v = np.gradient(c, t, axis=1)
        a = np.gradient(v, t, axis=1)
       
        return t, a, v, c

    def to_jointspace(self, coordinates: np.ndarray, tetherbot: TbTetherbot) -> np.ndarray:

        qs = np.empty((tetherbot.platform.m, coordinates.shape[1]))

        for i in range(coordinates.shape[1]):
            # inverse kinematics
            qs[:,i] = tetherbot.ivk(TransformMatrix(coordinates[:,i]))

        return qs
    
    def to_coordinatespace(self, qs: np.ndarray, tetherbot: TbTetherbot) -> np.ndarray:

        coordinates = np.zeros((6, qs.shape[1])) # column wise

        T0 = tetherbot.platform.T_world

        for i in range(qs.shape[1]) :
            # forward kinematics
            T0 = tetherbot.fwk(qs[:,i], T0)
            # use the result from the previous iteration as the guess for the next
               
            coordinates[:,i] = T0.decompose()

        return coordinates
        

class ProfileQArm(ProfileQPlatform):

    def to_jointspace(self, coordinates: np.ndarray, tetherbot: TbTetherbot) -> np.ndarray:
        
        qs = np.empty((tetherbot.platform.arm.dof, coordinates.shape[1])) # column-wise

        for i in range(coordinates.shape[1]):
            # inverse kinematics
            qs[:,i] = tetherbot.platform.arm.ivk(TransformMatrix(coordinates[:3,i]))

        return qs
    
    def to_coordinatespace(self, qs: np.ndarray, tetherbot: TbTetherbot) -> np.ndarray:
        
        coordinates = np.zeros((6, qs.shape[1]))

        for i in range(qs.shape[1]) :
            # forward kinematics
            coordinates[:,i] = tetherbot.platform.arm.fwk(qs[:,i]).decompose()

        return coordinates
    

class Profile():

    def __init__(self, t: np.ndarray, a: np.ndarray, v: np.ndarray, q: np.ndarray) -> None:
        
        self._coordinates  = q.T
        self._acceleration = a
        self._time         = t
        self._velocity     = v
        self._poses        = []

        for coordinate in self._coordinates:
            self._poses.append(TransformMatrix(coordinate))

    @property
    def coordinates(self) -> np.ndarray:

        return self._coordinates

    @property
    def acceleration(self) -> np.ndarray:

        return self._acceleration

    @property
    def velocity(self) -> np.ndarray:

        return self._velocity

    @property
    def time(self) -> np.ndarray:

        return self._time

    @property
    def poses(self) -> list[TransformMatrix]:

        return self._poses

    @property
    def length(self) -> int:

        return len(self._coordinates)

    def debug_plot(self, ax: plt.Axes = None, plot_poses: bool = False):

        # Plot x, y, z
        if ax is None:
            fig = plt.figure()
            ax  = fig.add_subplot(projection='3d') 

        ax.set_box_aspect([1,1,1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.plot(self._coordinates[:,0], self._coordinates[:,1], self._coordinates[:,2], color = "black")

        if plot_poses:
            for pose in self.poses:
                cframe(pose, parent = ax, scale = 0.5)
        
        plt.show()



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    flag = 0

    if flag == 0:
        from ..tools import tic, toc
        points = np.array([[1,4,6,6,6,6,6,7,5],
                           [1,2,3,3,3,3,3,3,-3],
                           [1,2,3,3,3,3,4,5,6],
                           [0,1,0,1,2,3,3,3,3]])
        points = np.array([[0,1,1,1,1],
                           [0,0,1,1,1],
                           [0,0,3,0,0],
                           [2,1,4,1,2]])

        tic()
        smoother   = NotSmoother()
        t, a, v, q = SlicedProfile6(a_t=[2,0.1], d_t=[1,0.2], v_t=[1,1], dt=0.05, smoother=smoother).calculate(points)
        toc()

        fig = plt.figure()

        axs = []
        axs.append(fig.add_subplot(231))
        axs.append(fig.add_subplot(232))
        axs.append(fig.add_subplot(233))
        axs.append(fig.add_subplot(234))
        axs.append(fig.add_subplot(235, projection='3d'))

        axs[0].set_title('time vs angle')   
        axs[0].plot(t,q[3])

        axs[1].set_title('time vs acceleration')   
        axs[1].plot(t,a[0])
        axs[1].plot(t,a[1])

        axs[2].set_title('time vs velocity')
        axs[2].plot(t,v[0])
        axs[2].plot(t,v[1])

        axs[3].set_title('time vs distance')
        axs[3].plot(t, q[0])
        axs[3].plot(t, q[1])
        axs[3].plot(t, q[2])

        axs[4].set_title('time vs distance')
        axs[4].plot(q[0], q[1], q[2], '-o')
        axs[4].plot(points[0], points[1], points[2])

        plt.show()
    
    if flag == 1:

        xyz = np.array([[0,1,2,3,4,5,6,7,8,9],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0]])

        t, a, v, q = Profile3(1, 1, 1, 0.5).calculate(xyz)     

        print(np.round(q,2))      

    
