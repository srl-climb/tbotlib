from __future__ import annotations
from abc        import ABC, abstractmethod, abstractstaticmethod
import numpy as np
import matplotlib.pyplot

def polysplinefit(y: np.ndarray, distance: float, resolution: float, limits: list = None, degree: int = 5):
    '''
    Creates a smooth quintic spline between the way points y. 
    The spline and its first and second derivatives are continous.
    Optional: Checks if the spline and its first and second derivatives is within limits.

    y:             way points of the segments, Note at least 2 way points are required
    distance:      distance between the way points
    resolution:    resolution with which the spline is sampled
    limits (opt.): lower and upper limits of the spline and its first and second derivatives
    degree:        degree of the spline (currenty supported: 3, 5)
    '''

    # limit the resolution
    if resolution >= len(y)*distance:
        return y, None, None, False

    # calculate 1st and 2nd derivatives the the points y
    n = len(y) 
    y_d1 = np.zeros(n)
    y_d2 = np.zeros(n)

    if n>2:
        #y_d1[1:-1] = ((y[1:-1] - y[:-2]) * (y[2:] - y[1:-1]) >= 0) * (y[2:] - y[:-2]) / (2 * distance)
        y_d1[1:-1] = (y[2:] - y[:-2]) / (2 * distance)
        #y_d2[1:-1] = ((y_d1[1:-1] - y_d1[:-2]) * (y_d1[2:] - y_d1[1:-1]) >= 0) * (y_d1[2:] - y_d1[:-2]) / (2 * distance)
        y_d2[1:-1] = (y_d1[2:] - y_d1[:-2]) / (2 * distance)

    # generate quintic spline
    if degree == 5:
        spline  = _SegmentedPoly5Spline(n-1, distance)
    elif degree == 3:
        spline = _SegmentedPoly3Spline(n-1, distance)

    x_0 = 0 
    x_1 = distance

    for i in range(n-1):
        
        p = spline.coefficients(x_0, x_1, y[i], y[i+1], y_d1[i], y_d1[i+1], y_d2[i], y_d2[i+1])

        # append to spline
        spline.add_segment(p)
        
        x_0 = x_0 + distance
        x_1 = x_1 + distance
    
    spline.evaluate(resolution)

    # check limits
    exitflag = True

    if limits is not None:
        if limits[0] is not None:
            if np.any(spline.y < limits[0][0]) or np.any(limits[0][1] < spline.y):
                exitflag = False
        if limits[1] is not None:
            if np.any(spline.y_d1 < limits[1][0]) or np.any(limits[1][1] < spline.y_d1):
                exitflag = False   
        if limits[2] is not None:
            if np.any(spline.y_d2 < limits[2][0]) or np.any(limits[2][1] < spline.y_d2):
                exitflag = False 

    return spline.x, spline.y, spline.y_d1, spline.y_d2, exitflag 

class _AbstractSegmentedPolySpline(ABC):

    def __init__(self, n: int, length: int, degree: int) -> None:   
        '''
        Spline consisting of n polynominal splines. 
        n:      number of segments
        length: length of a segment
        degree: degree of the spline
        '''

        self._n = n
        self._p = np.empty((n, degree+1)) 
        self._i = 0
        self.x: np.ndarray = None
        self.y: np.ndarray = None
        self.y_d1: np.ndarray = None
        self.y_d2: np.ndarray = None
        self._length = length

    def add_segment(self, p: np.ndarray) -> None:
        '''
        Add a segment.
        p: polynominal coeffecients (1d array)
        '''

        self._p[self._i, :] = p
        self._i = self._i + 1

    @abstractmethod
    def evaluate(self, dx: float) -> None:
        '''
        Evaluate the segmented spline in regular intervals.
        dx: size of each interval in x direction.
        '''
        pass
    
    @abstractstaticmethod
    def coefficients(self, *args):
        '''
        Calculate coefficients of a poly segment.
        '''
        pass

class _SegmentedPoly5Spline(_AbstractSegmentedPolySpline):

    def __init__(self, n: int, length: int) -> None:
        '''
        Spline consisting of n polynominal splines. 
        n:      number of segments
        length: length of a segment
        '''
        
        super().__init__(n, length, 5)
     
    @staticmethod
    def coefficients(x_0: float, x_1: float, y_0: float, y_1: float, y_d1_0: float, y_d1_1: float, y_d2_0: float, y_d2_1: float) -> np.ndarray:
        '''
        Calculate coefficients of a poly segment.
        x_0, x_1: x-values at start and end of segment.
        y_0, y_1: y-values at start and end of segment.
        y_d1_0, y_d1_1: values of the first derivative at start and end of segment.
        y_d2_0, y_d2_1: value of the second derivatives at start and end of segment.
        '''
        A = np.array([[1, x_0, x_0**2, x_0**3, x_0**4, x_0**5],
                      [1, x_1, x_1**2, x_1**3, x_1**4, x_1**5],
                      [0, 1, 2*x_0, 3*x_0**2, 4*x_0**3, 5*x_0**4],
                      [0, 1, 2*x_1, 3*x_1**2, 4*x_1**3, 5*x_1**4],
                      [0, 0, 2, 6*x_0, 12*x_0**2, 20*x_0**3],
                      [0, 0, 2, 6*x_1, 12*x_1**2, 20*x_1**3]])        
        b = np.array([y_0, y_1, y_d1_0, y_d1_1, y_d2_0, y_d2_1])

        # calculate coefficients of quintic spline
        return np.linalg.solve(A, b)

    def evaluate(self, dx: float):
        '''
        Evaluate the segmented spline in regular intervals.
        dx: size of each interval in x direction.
        '''

        self.x = np.linspace(0, self._n*self._length, int((self._n*self._length)//dx+1))
        self.y = np.empty(self.x.shape)
        self.y_d1 = np.empty(self.x.shape)
        self.y_d2 = np.empty(self.x.shape)
        
        # indices of the segments in which each x lies
        j = np.floor_divide(self.x, self._length).astype(int)
        j[-1] = np.clip(j[-1], None, self._n-1) # enures that the last point stays within the last segment 
        
        self.y = np.polynomial.polynomial.polyval(self.x, self._p[j,:].T, tensor=False)
        self.y_d1 = np.polynomial.polynomial.polyval(self.x, ([1,2,3,4,5]*self._p[j,1:]).T, tensor=False)
        self.y_d2 = np.polynomial.polynomial.polyval(self.x, ([2,6,12,20]*self._p[j,2:]).T, tensor=False)

class _SegmentedPoly3Spline(_AbstractSegmentedPolySpline):

    def __init__(self, n: int, length: int) -> None:
        '''
        Spline consisting of n polynominal splines. 
        n:      number of segments
        length: length of a segment
        '''
        
        super().__init__(n, length, 3)
     
    @staticmethod
    def coefficients(x_0: float, x_1: float, y_0: float, y_1: float, y_d1_0: float, y_d1_1: float, *_) -> np.ndarray:
        '''
        Calculate coefficients of a poly segment.
        x_0, x_1: x-values at start and end of segment.
        y_0, y_1: y-values at start and end of segment.
        y_d1_0, y_d1_1: values of the first derivative at start and end of segment.
        '''
        p_3 = (y_d1_0+y_d1_1-2*(y_1-y_0)/(x_1-x_0)) / (x_0-x_1)**2
        p_2 = (y_d1_1-y_d1_0) / 2*(x_1-x_0) - (3/2)*(x_0+x_1)*p_3
        p_1 = y_d1_0-3*x_0**2*p_3-2*x_0*p_2
        p_0 = y_0-x_0**3*p_3-x_0**2*p_2-x_0*p_1

        return np.array([p_0, p_1, p_2, p_3])
    
    def evaluate(self, dx: float):

        self.x = np.linspace(0, self._n*self._length, int((self._n*self._length)//dx+1))
        self.y = np.empty(self.x.shape)
        self.y_d1 = np.empty(self.x.shape)
        self.y_d2 = np.empty(self.x.shape)
        
        # indices of the segments in which each x lies
        j = np.floor_divide(self.x, self._length).astype(int)
        j[-1] = np.clip(j[-1], None, self._n-1) # enures that the last point stays within the last segment 

        self.y = np.polynomial.polynomial.polyval(self.x, self._p[j,:].T, tensor=False)
        self.y_d1 = np.polynomial.polynomial.polyval(self.x, ([1,2,3]*self._p[j,1:]).T, tensor=False)
        self.y_d2 = np.polynomial.polynomial.polyval(self.x, ([2,6]*self._p[j,2:]).T, tensor=False)