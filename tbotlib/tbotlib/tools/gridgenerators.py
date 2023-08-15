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