from __future__ import annotations
import numpy as np



class StructureMatrix:

    _H  = []  # Helper matrix to efficiently calculate the cross product
    _AT = []  # Structure matrix

    def __init__(self, L: np.ndarray = None, B: np.ndarray = None) -> None:
        
        self._H = np.array([[[ 0,  0,  0],
                             [ 0,  0,  1],
                             [ 0, -1,  0]],
                            [[ 0,  0, -1],
                             [ 0,  0,  0],
                             [ 1,  0,  0]],
                            [[ 0,  1,  0],
                             [-1,  0,  0],
                             [ 0,  0,  0]]])

        self._calculate(L, B)

    def __call__(self, L: np.ndarray = None, B: np.ndarray = None) -> np.ndarray:
        
        self._calculate(L, B)

        return self._AT

    @property
    def AT(self) -> np.ndarray:
        
        return self._AT

    def _calculate(self, L: np.ndarray, B: np.ndarray) -> None:

        if L is not None and B is not None:
            
            L = np.array(L)
            B = np.array(B)

            # Compute unit vectors be deviding each column vector of the cable vector by its length
            U = L / np.linalg.norm(L, axis=0)

            # Compute structure matrix/transpose of the Jacobian
            self._AT = np.zeros((L.shape[0]*2 , L.shape[1]))
            self._AT[:3,:] = U
            self._AT[3:,:] = np.einsum('ijk,ja,ka->ia', self._H, B, U)
            # same as cross(B, U, axis=0), but faster for smaller arrays

if __name__ == "__main__":
    
    L =  np.random.random(size=30).reshape(3,10)
    B =  np.random.random(size=30).reshape(3,10)
    AT = StructureMatrix() #m, n
    print(AT())
    print(AT(L,B))