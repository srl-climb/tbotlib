from __future__ import annotations
from typing import TYPE_CHECKING
from ..tools import lineseg_distance, tbbasefit, is_convex2, bbox_size
from ..fdsolvers import HyperPlaneShifting
from ..matrices import TransformMatrix
import numpy as np
import openGJK_cython as gjk
import itertools

if TYPE_CHECKING:
    from ..tetherbot import TbTetherbot, TbPart
    from .Graph import TbStepGraph
    from .Workspace import TbWorkspace
    from .Planner import PlanPlatform2Gripper, PlanPlatform2Hold


class _Feasibility:

    def eval(self, *args) -> bool:

        return bool


class TbFeasibility(_Feasibility):

    def eval(self, tetherbot: TbTetherbot) -> bool:

        pass

class TbWrenchFeasibility(TbFeasibility):

    def __init__(self, solver: HyperPlaneShifting = None, threshold: float = 0.0):

        if solver is None:
            self.solver = HyperPlaneShifting()
        else:
            self.solver = solver

        self.threshold = threshold

    def eval(self, tetherbot: TbTetherbot) -> bool:
        
        return self.solver.eval(tetherbot.W, tetherbot.F, tetherbot.tensioned)[0] >= self.threshold
    

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

            collisionfree = gjk.pygjk(collidable1.points_world, collidable2.points_world) > self.distance

            if not collisionfree:
                break

        return collisionfree

    
class TbWallPlatformCollisionFeasibility(TbCollisionFeasiblity):

    def eval(self, tetherbot: TbTetherbot) -> bool:

        f = self._gjk(tetherbot.platform, tetherbot.wall)

        return f


class TbGripperPlatformCollisionFeasibility(TbCollisionFeasiblity):

    def eval(self, tetherbot: TbTetherbot) -> bool:

        for gripper in tetherbot.grippers:

            collisionfree = self._gjk(tetherbot.platform, gripper)

            if not collisionfree:
                break

        return collisionfree


class TbTetherArmCollisionFeasibility(TbCollisionFeasiblity):

    def eval(self, tetherbot: TbTetherbot) -> bool:

        for tether in tetherbot.tethers:

            collisionfree = self._gjk(tetherbot.platform.arm.links[-1], tether)

            if not collisionfree:
                break

        return collisionfree
    

class StanceFeasibility(_Feasibility):

    def eval(self, u: tuple, tetherbot: TbTetherbot, graph: TbStepGraph) -> bool:

        pass


class StanceGeometricFeasibility(StanceFeasibility):

    def __init__(self, min_width: float = 0, max_width: float = np.inf):

        self.min_width = min_width
        self.max_width = max_width

    def eval(self, u: tuple, tetherbot: TbTetherbot, graph: TbStepGraph) -> bool:
        
        # check convexity and size of the stance, only necessary for full stances
        if -1 not in u:
            feasibility = False
            u = np.array(u)

            # stance polygon
            base = tbbasefit(tetherbot, output_format=1)
            stancepoly = base.inverse_transform(tetherbot.C_world[:, u[tetherbot.aorder.values()]], axis=0, copy=True).T[:,:2]  # stance polygon (gripper positions)

            if is_convex2(stancepoly, precision=6)[0]:
                min_width, max_width = bbox_size(stancepoly)
                if min_width > self.min_width and max_width < self.max_width:
                    feasibility = True
        else:
            feasibility = True
        
        return feasibility
    

class StanceWrenchFeasiblity(StanceFeasibility):

    def __init__(self, workspace: TbWorkspace):

        self.workspace = workspace

    def eval(self, u: tuple, tetherbot: TbTetherbot, graph: TbStepGraph) -> bool:

        feasibility = False

        # k-1 stance
        if -1 in u:

            # check for feasible pose using a workspace analysis
            grip_idx = u.index(-1)
            tetherbot.tension(grip_idx, False)
            feasibility, pose = self.workspace.calculate(tetherbot)
            tetherbot.tension(grip_idx, True)

            if feasibility:
                graph.set_reachable_pose(u, pose)

        # k stance
        else:
            # check for feasible poses in existing stances
            for i in range(tetherbot.k):
                utemp    = list(u)
                utemp[i] = -1
                utemp = tuple(utemp)

                if graph._graph.has_node(utemp) and graph.get_reachable(utemp):
                    tetherbot.platform.T_local = TransformMatrix(graph.get_reachable_pose(utemp))
                    feasibility = self.workspace.feasiblity.eval(tetherbot)

                    if feasibility:
                        break
            
            # check for feasible pose using a workspace analysis
            if not feasibility: 
                feasibility, pose = self.workspace.calculate(tetherbot)
                
                if feasibility:
                    graph.set_reachable_pose(u, pose)

        return feasibility
        

class StepFeasibility(_Feasibility):

    def eval(self, u: tuple, v: tuple, tetherbot: TbTetherbot, graph: TbStepGraph) -> bool:

        pass

class StepPathFeasibility(StepFeasibility):

    def __init__(self, platform2gripper: PlanPlatform2Gripper, platform2hold: PlanPlatform2Hold) -> None:
        
        self.platform2gripper = platform2gripper
        self.platform2hold = platform2hold

    def eval(self, u: tuple, v: tuple, tetherbot: TbTetherbot, graph: TbStepGraph) -> bool:
        
        # k-1 stance
        if -1 in u:
            # place
            grip_idx = u.index(-1)
            hold_idx = v[grip_idx]
            tetherbot.platform.T_world = TransformMatrix(graph.get_reachable_pose(u))
            tetherbot.tension(grip_idx, False)
            feasibility = self.platform2hold.plan(tetherbot, hold_idx, grip_idx)[2] is not None
            tetherbot.tension(grip_idx, True)
        # k stance
        elif -1 in v:     
            # pick
            grip_idx = v.index(-1)
            tetherbot.platform.T_world = TransformMatrix(graph.get_reachable_pose(v)) 
            tetherbot.tension(grip_idx, False)
            feasibility = self.platform2gripper.plan(tetherbot, grip_idx)[2] is not None
            tetherbot.tension(grip_idx, True)
        
        return feasibility


class StepDistanceFeasibility(StepFeasibility):

    def __init__(self, distance: float = 1.0) -> None:
        
        self.distance = distance

    def eval(self, u: tuple, v: tuple, tetherbot: TbTetherbot, graph: TbStepGraph) -> bool:
        
        # get full stance w
        if -1 in u:
            # place
            grip_idx = u.index(-1)
            w = v
        else:
            # pick
            grip_idx = v.index(-1)
            w = u 

        # triangle of the to be placed/picked gripper and its neighbours
        a = tetherbot.C_world[:, w[tetherbot.aorder[tetherbot.aorder.index(grip_idx)+1]]]
        b = tetherbot.C_world[:, w[tetherbot.aorder[tetherbot.aorder.index(grip_idx)-1]]]
        c = tetherbot.C_world[:, w[grip_idx]]

        # check step distance
        feasibility = lineseg_distance(c, a, b) <= self.distance

        return feasibility


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