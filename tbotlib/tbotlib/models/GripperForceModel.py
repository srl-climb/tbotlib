from __future__ import annotations
import numpy as np

def sec(x: float) -> float:

    return 1/np.cos(x)

class GripperForceModel:

    def eval(self, thetas: np.ndarray) -> np.ndarray:

        pass
    
    def debug_plot(self):

        import matplotlib.pyplot as plt

        thetas = np.linspace(0, 180, 100)
        fs = self.eval(thetas)

        np.float = float
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.plot(np.deg2rad(thetas), fs)
        ax.set_ylim(bottom = -1, top=40)
        plt.show()
    

class PanorelGripperForceModel(GripperForceModel):

    def __init__(self, fh_max: float = 18.0, fv_max: float = 28.0) -> None:
        
        self.fh_max = fh_max
        self.fv_max = fv_max
        
    def eval(self, thetas: np.ndarray) -> float:
        
        thetas = np.deg2rad(90 - thetas)
        
        h = thetas>=0
        f = np.hstack((np.minimum(self.fh_max / np.cos(thetas[h]), self.fv_max / np.sin(thetas[h])) , self.fh_max / np.cos(thetas[np.logical_not(h)])))

        return f 
    
if __name__ == '__main__':

    model = PanorelGripperForceModel()
    model.debug_plot()