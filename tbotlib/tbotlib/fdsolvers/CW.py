from __future__     import annotations
from .FD            import QuadraticProgram, ImprovedClosedMethod
from typing         import Union, Tuple
from scipy.spatial  import ConvexHull as qhull
from itertools      import combinations
import numpy      as np
#import tensorflow as tf

class Base():

    def __init__(self, m: int, n: int=6):

        self._m = m
        self._n = n
        

    def eval(self, AT: np.ndarray, W: np.ndarray, f_min: np.ndarray, f_max: np.ndarray, tensioned: np.ndarray) -> Tuple(bool, float):

        exitflag = False

        return 1, exitflag


class CornerCheck(Base):

    """
    Checks if a valid force distribution exists for a given wrench set
    AT          structure matrix
    W           wrench set or wrench acting on the robotic platform
    m           number of cables
    n           degrees of freedom
    f_min       vector of the minimum allowed cable force
    f_max       vector of the maximum allowed cable force
    exitflag    TRUE if a valid force distribution was found for each wrench
    """

    def __init__(self, m: int, n: int=6, fdsolver: Union[QuadraticProgram, ImprovedClosedMethod] = None):

        super().__init__(m, n)

        if fdsolver is None:
            self._fdsolver = QuadraticProgram(m, n)
        else:
            self._fdsolver = fdsolver


    def eval(self, AT: np.ndarray, W: np.ndarray, f_min: np.ndarray, f_max: np.ndarray, *_) -> Tuple(bool, float):

        W = np.atleast_2d(W)

        for w in W:
            _, exitflag = self._fdsolver.eval(AT, w, f_min, f_max)
            if exitflag == False:
                break

        return exitflag, exitflag


class QuickHull(Base):

    '''
    Capacity criterion based on 2014_Guay_Measuring: How Well a Structure Supports Varying External Wrenches
    '''

    def __init__(self, m: int, n: int=6):

        super().__init__(m, n)

        self._q     = 2**self._m                    # number of vertices in the feasible force set
        self._F     = np.empty((self._m, self._q))  # feasible force set, vertice-representation

    def eval(self, AT: np.ndarray, W_T: np.ndarray, f_min: np.ndarray, f_max: np.ndarray, *_) -> Tuple(bool, float):
        
        # calculate vertices of the feasible force set   
        for k in range(self._q):

            # convert k to binary representation
            k_beta = np.frombuffer(np.binary_repr(k, width=self._m).encode(), 'u1') - 48

            # vertices of the feasible force set
            self._F[:,k] = (np.eye(self._m) - np.diag(k_beta))@f_min + np.diag(k_beta)@f_max
        
        # project feasible force set to wrench space
        W_F = -AT@self._F
        
        # convex hull representation of the feasible force in the wrench space
        W_F = qhull(W_F.T, qhull_options='Qx').equations.T
        # Note: - input:  each column is a point
        #       - output: each column is normal, offset

        # calculate stability
        s = (W_T@W_F[:-1,:]-W_F[-1,:]) #/ np.linalg.norm(W_F[:-1,:], axis=0) not necessary
        
        if s.size == 0:
            print('Warning: zero size stability array')
            s = -1
        else:
            s = np.min(s)
       
        return s/1, s>=0


class HyperPlaneShifting(Base):
    '''
    Method based on Characterization of Parallel Manipulator Available Wrench Set Facets/ 
    On The Ability of a Cable-Driven Robot to Generate a Prescribed Set of Wrenches
    Capacity criterion based on 2014_Guay_Measuring: How Well a Structure Supports Varying External Wrenches
    '''

    def __init__(self, m: int, n: int=6):

        super().__init__(m, n)

        #tf.config.set_visible_devices([], 'GPU') # avoid using the GPU

        # possible combinations of n-1 columns of AT
        self._I = np.array(list(combinations(range(self._m), self._n-1)))   
        
        # helper variable: used to remove rows from U
        self._h1 = np.empty((self._n, self._n-1, self._n))
        for i in range(self._n):
            self._h1[i,:,:] = np.eye(self._n)[i != np.array(range(self._n)),:]

        # helper variable: used to create alternating signs
        self._h2 = np.empty((self._n,1))
        self._h2[::2]  = 1
        self._h2[1::2] = -1

    def eval(self, AT: np.ndarray, W_T: np.ndarray, f_min: np.ndarray, f_max: np.ndarray, *_) -> Tuple(bool, float):
        
        # linear independent combinations of AT
        I_0 = self._I[np.sum(np.linalg.svd(AT.T[self._I,:], compute_uv=False) > 1e-10, axis=1) == self._n-1]
        #I_0 = self._I[np.linalg.matrix_rank(AT.T[self._I,:]) == self._n-1]
        
        U   = AT[:, I_0].swapaxes(0,1)
        j   = U.shape[0]
        
        # compute normals
        N = np.empty((self._n,j))
        
        for i in range(self._n):
            N[i,:] = self._h2[i] * np.linalg.det(self._h1[i,:,:] @ U)

        N = N / np.linalg.norm(N, axis=0)
        
        # compute distances
        NTU = (N.T[:,np.newaxis] @ AT).reshape(-1,self._m)

        I_max = NTU>0
        I_min = NTU<0

        F_max = np.tile(f_max,(j,1))
        F_min = np.tile(f_min,(j,1))

        D_1 = + np.sum(F_max*NTU*I_max, axis=1) + np.sum(F_min*NTU*I_min, axis=1)
        D_2 = - np.sum(F_max*NTU*I_min, axis=1) - np.sum(F_min*NTU*I_max, axis=1)  
        
        # convex hull representation of the feasible force in the wrench space
        W_F              = np.empty((self._n+1,j*2))
        W_F[:self._n,:j] = -N
        W_F[:self._n,j:] =  N
        W_F[-1,:j]       =  D_2
        W_F[-1,j:]       =  D_1
        W_F              = -W_F 
        # Note: - input:  each column is a point
        #       - output: each column is normal, offset
        
        # calculate stability
        s = (W_T@W_F[:-1,:]-W_F[-1,:]) #/ np.linalg.norm(W_F[:-1,:], axis=0) not necessary

        if s.size == 0:
            print('Warning: zero size stability array')
            s = -1
        else:
            s = np.min(s)
        
        return s/1, s>=0


class AdaptiveCWSolver(Base):

    def __init__(self, m: int, n: int = 6):
        
        super().__init__(m, n)

        self._solvers: dict[int, HyperPlaneShifting] = {}

        for i in range(self._m+1):
            self._solvers[i] = HyperPlaneShifting(i, self._n)

    def eval(self, AT: np.ndarray, W: np.ndarray, f_min: np.ndarray, f_max: np.ndarray, tensioned: np.ndarray) -> Tuple(bool, float):
        
        m = np.sum(tensioned)

        return self._solvers[m].eval(AT[:,tensioned], W, f_min[tensioned], f_max[tensioned])


