from __future__  import annotations
from ..tools     import interweave
from abc         import ABC, abstractmethod
from scipy       import interpolate
import numpy as np


class AbstractSmoother(ABC):

    @abstractmethod
    def smooth(self, points: np.ndarray) -> np.ndarray:
        
        return points


class NotSmoother(AbstractSmoother):

    def smooth(self, points: np.ndarray) -> np.ndarray:
        
        return points


class ApproxSmoother(AbstractSmoother):

    def __init__(self, ds:float = 0.001, k: int = 3, s: float = 0) -> None:
        '''
        ds: approximate distance between the points on the spline 
        k:  degree of the spline 
        s:  smoothing condition   
        '''

        self.k  = k
        self.ds = ds
        self.s  = s

    def smooth(self, points: np.ndarray, axis: int = 1) -> np.ndarray:
        '''
        points: numpy array with points
        axis:   0: each row is a point
                1: each column is a point
        '''

        # ensure column wise points
        if axis == 0:
            points = points.T
        
        # ensure unique points
        points = np.unique(points, axis=1)

        # calculate spline parameter u
        l = np.sum(np.linalg.norm(np.diff(points), axis=0))    
        u = np.linspace(0, 1, int(l/self.ds))
        
        # degree of the spline
        k = np.clip(self.k, 0, points.shape[1]-1)
        
        # calculate spline
        if k >= 1:
            tck, _ = interpolate.splprep(points, k=k, s=self.s) 
            spline = np.array(interpolate.splev(u, tck))
        else:
            spline = np.array(points, copy=True)
       
        # restore shape
        if axis == 0:
            spline = spline.T

        return spline

        
class BsplineSmoother(AbstractSmoother):

    def __init__(self, ds: float = 0.001, k: int = 3, add_points: bool = False) -> None:
        '''
        ds:         approximate distance between the points on the spline       
        k:          degree of the spline
        add_points: add additional points between the control points  
        '''

        self.k          = k
        self.ds         = ds
        self.add_points = add_points

    def smooth(self, points: np.ndarray, axis: int = 1) -> np.ndarray:
        '''
        points: numpy array with points
        axis:   0: each row is a point
                1: each column is a point
        code is based on https://stackoverflow.com/a/35007804
        '''

        # ensure column wise points
        if axis == 0:
            points = points.T

        # calculate approximate length of the spline
        l = np.sum(np.linalg.norm(np.diff(points), axis=0))    

        # degree of the spline
        k = np.clip(self.k, 0, points.shape[1]-1)

        # add control points
        if k >= 2 and self.add_points:
            points = interweave(points, points[:,:-1] + (points[:,1:]-points[:,:-1])*0.5)

        # number of points
        n = points.shape[1]
        
        # knots of the spline
        knots = np.clip(np.arange(n+k+1)-k, 0, n-k)
        # creates duplicate knots at the edge of the spline to clamp down the spline at the endpoints

        # calculate and evaluate spline function g
        if k > 1:
            g = interpolate.BSpline(knots, points, k, axis=1)
            spline = g(np.linspace(0, n-k, int(l/self.ds)))
        else:
            spline = np.array(points, copy=True)

        # restore shape
        if axis == 0:
            spline = spline.T

        return spline
