from __future__ import annotations
import numpy as np

def circulargrid(radius: np.ndarray, angle: np.ndarray, z: float) -> np.ndarray:

    radius = np.array(radius)
    angle  = np.radians(np.array(angle))

    x = np.outer(np.cos(angle), radius).flatten()
    y = np.outer(np.sin(angle), radius).flatten()
    
    return np.array([x, y, np.ones(x.shape[0])*z])

def meshgrid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:

    return np.array(np.meshgrid(x, y, z, indexing='xy')).T.reshape(-1, 3)

def cylindricalgrid(x: np.ndarray, y: np.ndarray, radius: float):

    # create flat grid
    flatgrid = meshgrid(x, y, [0])[:,:2]

    # bend into cylinder where (0,0) of the flat grid is mapped to (0,0,0)
    angles = flatgrid[:,0] / radius
    x = radius * np.sin(angles)
    y = flatgrid[:,1]
    z = -np.sqrt((2*radius*np.sin(angles/2))**2 - x**2)

    angles = np.rad2deg(angles)

    return np.array([x, y, z, angles])

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    grid = cylindricalgrid([0,0.5,1,1.5,2,2.5], [0,0.5,1,1.5,2,2.5], 4)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.scatter(grid[0,:],grid[2,:])
    ax.add_patch(plt.Circle((0, -4), 4, ec=(0,0,0,1), lw=2, color='r'))
    

    plt.show()