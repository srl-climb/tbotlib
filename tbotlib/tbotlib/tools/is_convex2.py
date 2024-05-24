from __future__ import annotations
import numpy as np

def is_convex2(points: np.ndarray, precision: int = 12) -> tuple[bool, int]:
    '''
    Check if a list of 2d points form a convex polygon
    '''
    
    n = points.shape[0]     # number of vertices
    sign = 0                # unknown sign
    edgecount = 0           # actual number of edges, zero if not convex

    for i in range(n):

        # edges
        e1 = points[(i+1)%n] - points[i]
        e2 = points[(i+2)%n] - points[(i+1)%n]

        # if convex, all cross products must have the same sign!
        # zeros are ignored

        crossproduct = np.round(e1[0]*e2[1]-e2[0]*e1[1], precision)
        
        if crossproduct < 0:
            newsign = -1
            edgecount += 1
        elif crossproduct > 0:
            newsign = 1
            edgecount += 1
        else:
            # reject special case were edge is "folded back"
            if np.dot(e1, e2) < 0:
                return False, 0
            newsign = 0
        
        # if the sign changed (ignoring 0s), the polygon is concave
        if sign != 0 and newsign != 0 and newsign != sign:
            return False, 0
        
        # register first sign
        if sign == 0:
            sign = newsign

    return True, edgecount


if __name__ == "__main__":

    arr = -np.array([[0,0],[0,1],[1,1],[2,1],[2,0]])

    print(is_convex2(arr))

