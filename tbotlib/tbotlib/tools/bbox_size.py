from __future__ import annotations
import numpy as np

pi2 = np.pi/2

def bbox_size(points: np.ndarray) -> float:
    '''
    Find minimum width of the rotated bounding rectangle width using the rotating caliper algorithm
    points: Convex hull.
    '''

    edges = points[1:] - points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:,1], edges[:,0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    rotations = np.vstack([np.cos(angles), np.cos(angles-pi2),
                           np.cos(angles+pi2), np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotation to polygon
    points = np.dot(rotations, points.T)

    # find bounding points
    min_x = np.nanmin(points[:, 0], axis=1)
    max_x = np.nanmax(points[:, 0], axis=1)
    min_y = np.nanmin(points[:, 1], axis=1)
    max_y = np.nanmax(points[:, 1], axis=1)

    # calculate widths
    width_x = np.abs(max_x - min_x)
    width_y = np.abs(max_y - min_y)

    return np.min([width_x, width_y]), np.max([width_x, width_y])

if __name__ == "__main__":

    arr = -np.array([[0,0],[0,1],[1,1],[2,1],[2,0]])
    arr = np.array( ( (1,2), (5,4), (-1,-3) ))

    print(bbox_size(arr))