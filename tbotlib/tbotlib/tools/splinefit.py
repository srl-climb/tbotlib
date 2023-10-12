from __future__ import annotations
from abc        import ABC, abstractmethod, abstractstaticmethod
import numpy as np


class _SegmentedPoly3Spline():

    def __init__(self):

        pass

    def coefficients(self, x_0: float, x_1: float, y_0: float, y_1: float, yd1_0: float, yd1_1: float) -> np.ndarray:
        '''
        Calculate coefficients of n-1 poly spline segments
        x_0, x_1: x-values at start and end of the segments (1xn-1 array)
        y_0, y_1: y-values at start and end of the segments (dxn-1 array)
        yd1_0, yd1_1: values of the first derivative at start and end of the segments (dxn-1 array)
        '''
        p_3 = (yd1_0+yd1_1-2*(y_1-y_0)/(x_1-x_0)) / (x_0-x_1)**2
        p_2 = (yd1_1-yd1_0) / (2*(x_1-x_0)) - (3/2)*(x_0+x_1)*p_3
        p_1 = yd1_0-3*x_0**2*p_3-2*x_0*p_2
        p_0 = y_0-x_0**3*p_3-x_0**2*p_2-x_0*p_1
        
        return np.array([p_0, p_1, p_2, p_3])
    
    def evaluate(self, x: np.ndarray, y: np.ndarray, spline_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Evaluate multiple poly splines with n-1 segments
        x: Array with x-values of the way points between the segments (1xn array)
        y: Arrays with y-values of the way points between the segmetns (dxn array)
        spline_x: Array with x values to evaluate the poly splines at (1xn array)
        '''
        
        # gradient of the input points
        yd1: np.ndarray = np.gradient(y, x, axis=1)
        
        # polynomical coefficents for each segment of x and y (coeffients; row of y; segment)
        p = self.coefficients(x[:-1], x[1:], y[:,:-1], y[:,1:], yd1[:,:-1], yd1[:,1:])
        # indices of the segments in which each spline_x lies
        i = np.clip(np.searchsorted(x, spline_x, side='right')-1, 0, x.shape[0]-2)
        
        # evaluate each spline
        spline_y = np.polynomial.polynomial.polyval(spline_x, p[:,:,i], tensor=False)
        spline_yd1 = np.gradient(spline_y, spline_x, axis=1)
        spline_yd2 = np.gradient(spline_yd1, spline_x, axis=1)

        return spline_y, spline_yd1, spline_yd2


def polysplinefit(y: np.ndarray, step: float, spline_dx: float, spline_xmax: float = None, spline_limits: np.ndarray = None, spline_degree: int = 3, iter_max: int = 0, mode: str = 'whole') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    '''
    Creates and evaluates d segmented poly splines between n way points which satisfy the spline limits.
    The x-values for the first iteration are automatically calculated by dividing spline_xmax into n equidistant segments.
    The x-distance of the segments is increased iteratively until the limits are satisfied.
    y:              way points of the poly spline segments (dxn array), at least 2 way points are required
    step:           value with which the x-distance between the y-points is increased every iteration, if the limits are not met
    spline_limits:  lower and upper limits for the spline y-values and its 1st and 2nd derivative (3x2 array or 1x3x2 array or dx3x2 array or None), 
                    if only a single set of limits is specified the same limits are applied to each spline
    spline_xmax     maximum x-value of the spline in the first iteration
                    if provided the x-values of the way points are arranged equidistant between 0 and spline_xmax
                    otherwise, the x-distances of the way points are estimated based on the y-distance and provided limits of the first derivative
    spline_dx:      distance between x-values of the spline
    spline_degree:  degree of the spline 
    iter_max:       maximum number of iterations
    mode:           'whole': If the limits in any segment are not satisfied the dx of each segment is increased by step, 'segmental': If the limits are not satisfied the dx of the segment is increased by step
    Returns
    spline_x:   x-values of the spline (1xm array)
    spline_y:   y-values of the spline (dxm array)
    spline_yd1: first derivative of the spline (dxm array)
    spline_yd2: second derivative of the spline (dxm array)
    exitflag:   Exitflag: (0: maximum number of iterations reached before spline_limits were satisfied, 1: spline_limits and spline_xmax were satisfied, >1: spline_limits were satisfied after n-iterations
    '''

    # default spline limits are infinite
    if spline_limits is None:
        spline_limits = np.array(((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)))

    # parse spline limits into correct shape
    if spline_limits.ndim == 2: 
        spline_limits = spline_limits[None, :, :]
    if spline_limits.shape[0] != y.shape[0]:
        spline_limits = np.repeat(spline_limits, y.shape[0], axis=0)

    # result if only one way point was given
    if y.shape[1] < 2:
        return np.zeros(1), y, np.zeros(y.shape), np.zeros(y.shape), True

    # automatically calculate x-distances between way points for first iteration
    if spline_xmax is None:
        dx = np.max(np.abs(np.diff(y, axis=1) / np.min(np.abs(spline_limits[:,1,:]), axis=1)[:, None]) , axis=0) * 1
    else:
        dx = np.ones(y.shape[1]-1) * (spline_xmax/(y.shape[1]-1))
    
    # pre allocate array for x-values of the way points
    x = np.zeros(y.shape[1])
    
    # create poly spline object
    if spline_degree == 3:
        spline = _SegmentedPoly3Spline()
    else:
        raise NotImplementedError('Degree ' + str(spline_degree) + ' not implemented')

    # iteration counter
    counter = 0

    # spline fitting loop
    while True:
        counter = counter+1

        # calculate x-values of the way points
        x[1:] = np.cumsum(dx)
        # calculate x-values of the spline, as x[-1] changes it can not be preallocated
        spline_x = np.arange(0, x[-1], spline_dx)
        # evaluated segmented spline
        spline_y, spline_yd1, spline_yd2 = spline.evaluate(x, y, spline_x)
        # check limits
        inlimit = np.all((spline_limits[:,0,0, None] <= spline_y) & (spline_y <= spline_limits[:,0,1, None]),axis=0) & \
                  np.all((spline_limits[:,1,0, None] <= spline_yd1) & (spline_yd1 <= spline_limits[:,1,1, None]),axis=0) & \
                  np.all((spline_limits[:,2,0, None]<= spline_yd2) & (spline_yd2 <= spline_limits[:,2,1, None]),axis=0)
        
        # exit conditions
        if counter == 1 and np.all(inlimit):
            # spline_limits and spline_xmax were satisfied
            exitflag = 1
            break
        elif np.all(inlimit):
            # spline_limits were satisfied after n-iterations
            exitflag = counter
            break
        elif iter_max <= counter:
            # maximum number of iterations reached before spline_limits were satisfied
            exitflag = 0
            break

        if mode == 'whole':
            # increase x-distance of every segment
            dx = dx + step
        elif mode == 'segmental':
            # find segments whith splines which do not satisfy limits
            i = np.clip(np.searchsorted(x, spline_x, side='right')-1, 0, x.shape[0]-2) # segment index of each spline x-value
            j = np.unique(i[np.logical_not(inlimit)]) # segment indices with splines which do not satisfy the limits
            # increase x-distance of segments with splines which do not satisfy the limits
            dx[j] = dx[j] + step
        else:
            raise NotImplementedError('Mode ' + str(mode) + ' not implemented')

    return spline_x, spline_y, spline_yd1, spline_yd2, exitflag


""" def polysplinefit(y: np.ndarray, distance: float, resolution: float, limits: list = None, degree: int = 5):
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
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b)[0]

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
        p_2 = (y_d1_1-y_d1_0) / (2*(x_1-x_0)) - (3/2)*(x_0+x_1)*p_3
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
        self.y_d2 = np.polynomial.polynomial.polyval(self.x, ([2,6]*self._p[j,2:]).T, tensor=False) """