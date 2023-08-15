from __future__ import annotations
import numpy as np

def set_axes_equal(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, a: float = 0) -> None:
    
    x_mid = (x.min() + x.max())/2
    y_mid = (y.min() + y.max())/2
    z_mid = (z.min() + z.max())/2

    ax.set_xlim(x_mid+a, x_mid-a)
    ax.set_ylim(y_mid+a, y_mid-a)
    ax.set_zlim(z_mid+a, z_mid-a)

    ax.set_box_aspect((1,1,1)) 