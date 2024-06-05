from __future__ import annotations
from qpsolvers  import solve_qp
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..tetherbot import TbTetherForceSet

class Base():

    def __init__(self):

        self._n         = 6
        self._check_res = False

    def eval(self, w: np.ndarray, F: TbTetherForceSet) -> tuple[np.ndarray, int]:

        f        = np.zeros((1, F.m()))
        exitflag = 0

        return f, exitflag

    def check(self, w: np.ndarray, F: TbTetherForceSet, f: np.ndarray, exitflag: int) -> tuple[np.ndarray, int]:

        # Check residual if desired and the results were valid
        if self._check_res == True and exitflag == 1:
            
            # Check residual
            res = np.linalg.norm(np.matmul(F.AT(), f) + w)
            tf  = np.round(res, 4) == 0

            if tf == 0:
                print("Large residual: " + str(res))
                exitflag = 0
                f = []

        return f, exitflag


class QuadraticProgram(Base):

    """
    Quadratic programming solver for tether climbing robots
    
    Quadratic program:
                                    Gx<=h
    min 1/2*x^T*P*x+q^t*x such that A*x=b
                                    lb<=x<=ub
    """
    
    def eval(self, w: np.ndarray, F: TbTetherForceSet) -> tuple[np.ndarray, int]:

        # x = solve_qp(P, q, G, h, A, b)
        m = F.m()
        P = np.eye(m)
        q = np.zeros(m)
        G = F.halfspaces()[:, :-1]
        h = -F.halfspaces()[:, -1]
        A = F.AT()
        b = -w
        f = solve_qp(P, q, G, h, A, b, solver = 'quadprog')

        if f is None:
            exitflag = 0
        else:
            f = np.round(f, 6)
            exitflag = 1

        return self.check(w, F, f, exitflag)    


class ImprovedClosedMethod(Base):

    """
    Applies the Imporved Closed Method to find a force distribution of a
    cable-driven robot

    AT          structure matrix
    w           wrench acting on the robotic platform
    m           number of cables
    n           degrees of freedom
    f_min       vector of the minimum allowed cable force
    f_max       vector of the maximum allowed cable force
    f           vector of the cable force
    exitflag    exit condition
                   1: Function converged to the olution f
                   0: No solution found
    """

    def eval(self, w: np.ndarray, F: TbTetherForceSet) -> tuple[np.ndarray, int]:

        m = F.m()
        
        f_max = F.f_max()
        f_min = F.f_min()
        
        AT = F.AT()

        # Redundancy
        r = m - self._n

        # Helper matrix
        H = np.vstack((f_min, f_max))

        # Medium feasible cable force distribution
        f_m = 0.5 * (f_min + f_max)

        # Preallocate force distribution vecotrs
        f_v = np.zeros(m)
        f   = np.zeros(m)
        
        # Active columns/elements
        i = np.array(range(m))
        
        for j in range(m):
            
            # Compute variable part of the force distribution
            f_v[i] = np.matmul(-np.linalg.pinv(AT[:,i]), w + np.matmul(AT[:,i], f_m[i]))

            # Compute the force distribution
            f[i] = np.around(f_m[i] + f_v[i], 5)
            
            # Solution found
            if np.all(f_min <= f) and np.all(f <= f_max):
                exitflag = 1
                break

            # No solution found
            elif np.any(np.linalg.norm(f_v[i]) > 0.5 * np.sqrt(m) * (f_max[i] - f_min[i])) or r < j:
                exitflag = 0
                break
            
            # Find h which is the k-th cable with the largest force over f_max or under f_min
            (h, k) = self.maxsub(np.vstack((f_min - f, f - f_max)))
            
            # Set the cable force of the k-the cable to the maximum or minimum          
            f[k] = H[h,k]

            # Reduce the order of the problem by dropping the k-th column/element
            i = i[i!=k]
            w = f[k] * AT[:, k] + w

        return self.check(AT, w, f, exitflag)  

    @staticmethod
    def maxsub(M: np.ndarray) -> tuple[int, int]:

        """
        Returns the subscript of the maximum element in a matrix
        """
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
        (row, col) = np.unravel_index(np.argmax(M, axis=None), M.shape)
        
        return row, col