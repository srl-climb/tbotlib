# Source
# https://stackoverflow.com/questions/54442057/calculate-the-euclidian-distance-between-an-array-of-points-to-a-line-segment-in/54442561#54442561
# https://stackoverflow.com/a/56467661

import numpy as np

def lineseg_distance(p, a, b):

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p - a, d)
    
    return np.hypot(h, np.linalg.norm(c))

if __name__ == '__main__':

    a = np.array([0,0,0])
    b = np.array([1,0,0])

    print('point on line')
    print(lineseg_distance(np.array([0,0,0]),a,b))

    print('point on line but ouside segment')
    print(lineseg_distance(np.array([99,0,0]),a,b))

    print('point in segment on distance')
    print(lineseg_distance(np.array([0.5,1,0]),a,b))

    print('point in segment on distance')
    print(lineseg_distance(np.array([0.1,1,0]),a,b))

    print('last test')
    a = np.array([0.98961584,  0.,         -0.12435031])
    b = np.array([0.49869893,  2.5 ,       -0.03120933])
    c = np.array([ 0.98961584 , 0.  ,       -0.12435031])
    print(lineseg_distance(c,a,b))
