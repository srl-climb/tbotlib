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

        plt.plot(thetas, fs)
        plt.ylim(bottom = -1, top=40)
        plt.show()
    

class PanorelGripperFoceModel(GripperForceModel):

    def __init__(self, kappa_1: float = 18.0, kappa_2: float = 28.0) -> None:
        
        self.kappa_1 = kappa_1
        self.kappa_2 = kappa_2
        
    def eval(self, thetas: np.ndarray) -> float:
        
        thetas = np.deg2rad(90 - thetas)
        
        h = thetas>=0
        f = np.hstack((np.minimum(self.kappa_1 / np.cos(thetas[h]), self.kappa_2 / np.sin(thetas[h])) , self.kappa_1 / np.cos(thetas[np.logical_not(h)])))

        return f 
    
if __name__ == '__main__':

    model = PanorelGripperFoceModel()
    model.debug_plot()