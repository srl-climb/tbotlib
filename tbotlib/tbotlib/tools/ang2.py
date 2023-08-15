from __future__ import annotations
import numpy as np

def ang2(u: np.ndarray, v: np.ndarray) -> float:


    print(u@v / np.linalg.norm(u)/ np.linalg.norm(v))

    theta = np.arccos(np.clip( u@v /np.linalg.norm(u)/np.linalg.norm(v), -1, 1))

    return theta

if __name__ == '__main__':

    u = np.array((0,1))
    v = np.array((1,0))

    print(ang2(u, v))