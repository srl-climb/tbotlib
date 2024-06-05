from __future__     import annotations
from .FD            import QuadraticProgram, ImprovedClosedMethod
from typing         import Union, Tuple, TYPE_CHECKING
from scipy.spatial  import ConvexHull as qhull
from itertools      import combinations
import numpy      as np

if TYPE_CHECKING:
    from ..tetherbot    import TbWrenchSet, TbTetherForceSet, TbPolytopeWrenchSet

class Base():

    def __init__(self):

        self._n = 6

    def eval(self, W: TbWrenchSet, F: TbTetherForceSet) -> Tuple[float, bool]:

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

    def __init__(self, fdsolver: Union[QuadraticProgram, ImprovedClosedMethod] = None):

        super().__init__()

        if fdsolver is None:
            self._fdsolver = QuadraticProgram()
        else:
            self._fdsolver = fdsolver


    def eval(self, W: TbPolytopeWrenchSet, F: TbTetherForceSet, *_) -> Tuple[float, bool]:

        for w in W.vertices_world:
            _, exitflag = self._fdsolver.eval(F, w)
            if exitflag == False:
                break

        return exitflag, exitflag


class QuickHull(Base):

    '''
    Capacity criterion based on 2014_Guay_Measuring: How Well a Structure Supports Varying External Wrenches
    '''

    def eval(self, W: TbWrenchSet, F: TbTetherForceSet, *_) -> Tuple[float, bool]:
        
        # project vertices of feasible force set to wrench space
        W_F = -F.AT() @ F.vertices().T
        
        # convex hull representation of the feasible force in the wrench space
        W_F = qhull(W_F.T, qhull_options='Q14').equations.T
        # Note: - input:  each column is a point
        #       - output: each column is normal, offset

        # calculate stability
        s = W.hdistance(W_F[:-1,:], W_F[-1,:])
        
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

    def __init__(self):

        super().__init__()

        # cache variable for possible combinations of n-1 columns of AT
        self._combinations: dict[int, np.ndarray] = {}   
        
        # helper variable to remove rows from U
        self._h1 = np.empty((self._n, self._n-1, self._n))
        for i in range(self._n):
            self._h1[i,:,:] = np.eye(self._n)[i != np.array(range(self._n)),:]

        # helper variable to create alternating signs
        self._h2 = np.empty((self._n,1))
        self._h2[::2]  = 1
        self._h2[1::2] = -1

    def combinations(self, m: int) -> np.ndarray:

        if m not in self._combinations:
            self._combinations[m] = np.array(list(combinations(range(m), self._n-1)))
        
        return self._combinations[m]

    def eval(self, W: TbWrenchSet, F: TbTetherForceSet, *_) -> Tuple[float, bool]:

        # structure matrix
        AT = F.AT()
        m  = F.m()

        # possible combinations of n-1 columns of AT
        I = self.combinations(m)
        
        # linear independent combinations of AT
        I_0 = I[np.sum(np.linalg.svd(AT.T[I,:], compute_uv=False) > 1e-10, axis=1) == self._n-1]
        
        U   = AT[:, I_0].swapaxes(0,1)
        j   = U.shape[0]
        
        # compute normals
        N = np.empty((self._n,j))
        
        for i in range(self._n):
            N[i,:] = self._h2[i] * np.linalg.det(self._h1[i,:,:] @ U)

        N = N / np.linalg.norm(N, axis=0)
        
        # compute distances
        NTU = (N.T[:,np.newaxis] @ AT).reshape(-1, m)

        I_max = NTU>0
        I_min = NTU<0

        F_max = np.tile(F.f_max(),(j,1))
        F_min = np.tile(F.f_min(),(j,1))

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
        s = W.hdistance(W_F[:-1,:], W_F[-1,:])

        if s.size == 0:
            print('Warning: zero size stability array')
            s = -1
        else:
            s = np.min(s)

        return s/1, s>=0