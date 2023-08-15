import numpy as np

def ang3(u: np.ndarray, v: np.ndarray, output: bool = 0) -> float:

    '''
    Returns angle between vectors u and v.
    u, v:   Vectors
    output: Specifies output type (0: degrees, 1: radians)
    '''

    # see https://stackoverflow.com/a/2827466/425458
    c = np.dot(u/np.linalg.norm(u), v/np.linalg.norm(v))
    print(c)
    if output:
        return np.arccos(np.clip(c, -1, 1))
    else:
        return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


if __name__ == '__main__':

    u = np.array([0,1])
    v = np.array([0,2])

    c = ang3(u,v,0)

    print(c)
