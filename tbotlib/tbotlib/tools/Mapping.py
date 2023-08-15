from __future__ import annotations
import operator
import numpy as np


class Mapping():

    def __init__(self, a_to_b: np.ndarray) -> None:

        self._a_to_b = np.array(a_to_b)                     # anchor point a and b ot the tethers
        self._m      = len(self._a_to_b)                    # number of tethers
        self._k      = np.max(self._a_to_b, axis = 0)[0]    # number of anchor points a
        self._l      = np.max(self._a_to_b, axis = 0)[1]    # number of anchor points b 

        assert self._a_to_b.shape[1] == 2, 'Each tether must have two anchorpoints'
        assert self._iscomplete(self._a_to_b[:,0]), 'Anchorpoint a index missing'
        assert self._iscomplete(self._a_to_b[:,1]), 'Anchorpoint b index missing'

        self._from_a = self._from(self._a_to_b[:,0])    # tethers of anchor points a
        self._from_b = self._from(self._a_to_b[:,1])    # tethers of anchor points b

    @property
    def a_to_b(self) -> np.ndarray:

        return self._a_to_b

    @property
    def from_a(self) -> dict[int, list[int]]:

        return self._from_a

    @property
    def from_b(self) -> dict[int, list[int]]:

        return self._from_b

    @property
    def m(self) -> int:

        return self._m

    @property
    def k(self) -> int:

        return self._k

    @property
    def l(self) -> int:

        return self._l
    
    def _iscomplete(self, u: np.ndarray) -> bool:
        '''
        Check if u contains all indices between 0 and max(u)
        u: 1D array
        '''
        return np.all(np.in1d(np.arange(np.max(u)), u))

    def _from(self, u: np.ndarray) -> dict[int, list[int]]:
        '''
        Create dictionary v
            key:   values of u 
            value: indices of the values in u
        u: 1D array
        v: dicionary
        '''

        u = tuple(u)
        v: dict[int, list[int]] = {}

        for i in range(len(u)):
            
            if u[i] not in v:
                v[u[i]] = []
            
            v[u[i]].append(i)

        return dict(sorted(v.items(), key=operator.itemgetter(0))) # sort based on key


if __name__ == "__main__":

    M = Mapping([[1,2],[3,4],[0,2],[2,0],[1,1],[0,3]])

    print(M.a_to_b)
    print(M.from_a)
    print(M.from_b)
