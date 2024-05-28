from __future__ import annotations
from typing import TYPE_CHECKING
from math import sqrt

from tbotlib.tetherbot import TbTetherbot
from ..tools import ang3, lineseg_distance
from ..fdsolvers     import AdaptiveCWSolver, CW
import numpy as np
import openGJK_cython as gjk
import itertools



if TYPE_CHECKING:
    from ..tetherbot import TbTetherbot, TbPart
    from ..matrices import TransformMatrix

class _Feasibility:

    def eval(self, *args) -> bool:

        return bool


class TbFeasibility(_Feasibility):

    def eval(self, tetherbot: TbTetherbot) -> bool:

        pass

class TbWrenchFeasibility(TbFeasibility):

    def __init__(self, m: int = 10, n: int = 6, solver: AdaptiveCWSolver = None, threshold: float = 0.0):

        if solver is None:
            self.solver = AdaptiveCWSolver(m, n)
        else:
            self.solver = solver

        self.threshold = threshold

    def eval(self, tetherbot: TbTetherbot) -> bool:

        return self.solver.eval(tetherbot.AT, tetherbot.W, tetherbot.f_min, tetherbot.f_max, tetherbot.tensioned)[1] >= self.threshold
    

class TbTetherLengthFeasibility(TbFeasibility):

    def eval(self, tetherbot: TbTetherbot) -> bool:
        
        return all(tetherbot.l_min < tetherbot.l) and all(tetherbot.l < tetherbot.l_max)


class TbJointLimitFeasibility(TbFeasibility):

    def eval(self, tetherbot: TbTetherbot) -> bool:
        
        return tetherbot.platform.arm.valid()
    

class TbGripperPlatformDistanceFeasibility(TbFeasibility):

    def __init__(self, distance: float = 0.26) -> None:
        
        self.distance = distance

    def eval(self, tetherbot: TbTetherbot) -> bool:

        return all(np.linalg.norm(tetherbot.A_world - tetherbot.platform.r_world[:,None], axis=0) >= self.distance)
    

class TbArmTetherDistanceFeasibility(TbFeasibility):

    def __init__(self, distance: float = 0.1) -> None:

        self.distance = distance

    def eval(self, tetherbot: TbTetherbot) -> bool:

        feasible = False
        
        for tether, tensioned in zip(tetherbot.tethers, tetherbot.tensioned):
            if tensioned:
                feasible =  lineseg_distance(tetherbot.platform.arm.links[-1].r_world, 
                                             tether.anchorpoints[0].r_world, 
                                             tether.anchorpoints[1].r_world) >= self.distance
                
                if not feasible:
                    break

        return feasible


class TbCollisionFeasiblity(TbFeasibility):

    def __init__(self, distance: float = 0.1):

        self.distance = distance

    def eval(self, tetherbot: TbTetherbot) -> bool:

        return False

    def _gjk(self, part1: TbPart, part2: TbPart):

        for collidable1, collidable2 in itertools.product(part1.collidables, part2.collidables):

            collisionfree = gjk.pygjk(collidable1.points_world, collidable2.points_world) >= self.distance

            if not collisionfree:
                break

        return collisionfree

    
class TbWallPlatformCollisionFeasibility(TbCollisionFeasiblity):

    def eval(self, tetherbot: TbTetherbot) -> bool:

        return self._gjk(tetherbot.platform, tetherbot.wall)


class StanceFeasibility(_Feasibility):

    def eval(u: tuple):

        pass

class StanceConvexityFeasibility(StanceFeasibility):

    pass

class StanceWidthFeasibility(StanceFeasibility):

    pass

class StepFeasibility(_Feasibility):

    def eval(u: tuple, v: tuple):

        pass

class StepWidthFeasibility(StepFeasibility):

    pass


class FeasibilityContainer(_Feasibility):

    def __init__(self, feasibilities: list[_Feasibility] = None) -> None:
        
        if feasibilities is None:
            self.feasibilities = []
        else:
            self.feasibilities = feasibilities

    def eval(self, *args):

        feasible = False

        for feasibility in self.feasibilities:
            
            feasible = feasibility.eval(*args)

            if not feasible:
                break
        
        return feasible
    
    def add(self, value: _Feasibility) -> None:

        self.feasibilities.append(value)


if __name__ == "__main__":

    f: FeasibilityContainer[TbFeasibility] = FeasibilityContainer()
    f.eval()

    list()