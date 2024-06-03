from __future__     import annotations
from ..tetherbot    import TbTetherbot
from ..tools        import basefit, ang3, uniqueonly, tic, toc, tbbasefit
from ..matrices     import TransformMatrix, NdTransformMatrix
from .Workspace     import TbWorkspace
from .Path          import Path6, ClimbPath
from .Metric        import L1Metric, TbAlignmentMetric, StanceDisplacementMetric, ConstantMetric, _Metric
from .Feasibility   import FeasibilityContainer, TbWrenchFeasibility, TbJointLimitFeasibility
from typing         import Tuple, List, TYPE_CHECKING
from abc            import ABC, abstractmethod
from heapq          import heappop, heappush
from copy           import deepcopy
from math           import sqrt
import numpy        as np
import networkx     as nx

if TYPE_CHECKING:
    from .Planner   import PlanPlatform2Hold, PlanPlatform2Gripper

class SearchGraph(ABC):

    def __init__(self, iter_max: int = 5000, auto_clear: bool = True) -> None:
        
        self._graph      = nx.Graph()
        self._iter_max   = iter_max     # max iteration limit for search
        self._auto_clear = auto_clear   # automatically clear graph at the beginning of each search call

    def get_cost(self, u: Tuple, v: Tuple) -> float:

        return self._graph.edges[u, v]['cost']

    def get_heuristic(self, u: Tuple) -> float:

        return self._graph.nodes[u]['heuristic']

    def get_reachable(self, u: Tuple) -> float:

        return self._graph.nodes[u]['reachable']
    
    def set_reachable(self, u: Tuple, value: float) -> None:

        self._graph.nodes[u]['reachable'] = value

    def get_traversable(self, u: Tuple, v: Tuple) -> float:

        return self._graph.edges[u, v]['traversable']
    
    def set_traversable(self, u: Tuple, v: Tuple, value: float) -> None:

        self._graph.edges[u, v]['traversable'] = value

    def get_fscore(self, u: Tuple) -> float:

        return self._graph.nodes[u]['fscore']

    def set_fscore(self, u: Tuple, value: float) -> None:

        self._graph.nodes[u]['fscore'] = value

    def get_gscore(self, u: Tuple) -> float:

        return self._graph.nodes[u]['gscore']

    def set_gscore(self, u: Tuple, value: float) -> None:

        self._graph.nodes[u]['gscore'] = value

    def get_neighbours(self, u: Tuple) -> List[Tuple]:

        # list of nodes to which an edge exist from u
        neighbours = []

        for neighbour in self._get_potential_neighbours(u):
            
            # if the neighbour hasn't been visited yet, add the node to the graph
            if not self._graph.has_node(neighbour):

                self.add_node(neighbour)

            # if an edge to the neighbour hasn't beed visited yet, add it to the graph
            if not self._graph.has_edge(u, neighbour) and self.get_reachable(neighbour) > 0:
    
                self.add_edge(u, neighbour)

            # if the neighbour is reachable and the edge to is is traversable, add it to the neighbour list
            if self.get_reachable(neighbour) > 0 and self.get_traversable(u, neighbour) > 0:
            
                neighbours.append(neighbour)

        return neighbours

    def add_node(self, u: Tuple) -> None:

        heuristic = self._calc_heuristic(u)
        reachable = self._calc_reachable(u)

        self._graph.add_node(tuple(u), heuristic = heuristic, reachable = reachable, gscore = 0, fscore = 0)
        # set gscore/fscore to 0 as default values for the A-star search

    def add_edge(self, u: Tuple, v: Tuple) -> None:

        cost        = self._calc_cost(u, v)
        traversable = self._calc_traversable(u, v)

        self._graph.add_edge(u, v, cost = cost, traversable = traversable)

    @abstractmethod
    def _calc_cost(self, u: Tuple, v: Tuple) -> float:

        pass

    @abstractmethod
    def _calc_heuristic(self, u: Tuple) -> float:

        pass

    @abstractmethod
    def _calc_reachable(self, u: Tuple) -> bool:
        
        return True

    @abstractmethod
    def _calc_traversable(self, u: Tuple, v: Tuple) -> bool:

        return True

    @abstractmethod
    def _get_potential_neighbours(self, u: Tuple) -> List[Tuple]:

        pass

    @abstractmethod
    def is_goal(self, u: Tuple) -> bool:

        return True

    def search(self, start: Tuple, goal: Tuple) -> list[tuple[float]]: 
        
        # ASTAR SEARCH ALGORITHM
        # implementation based on https://code.activestate.com/recipes/578919-python-a-pathfinding-with-binary-heap/
        self._start = start
        self._goal  = goal

        # prepare graph
        if self._auto_clear:
            self._graph.clear()
        self.add_node(self._start)
        self.add_node(self._goal)

        # prepare variables
        camefrom = {}
        closed   = set()
        open     = []
        iter     = 0
        
        self.set_gscore(self._start, 0)
        self.set_fscore(self._start, self._calc_heuristic(self._start))
       
        heappush(open, (self.get_fscore(self._start), self._start)) 

        while open:

            # iteration maximum reached?
            if iter >= self._iter_max:
                break
            iter += 1
            
            # remove item with the smallest fscore from the heap
            current = heappop(open)[1]    

            # goal reached?
            if self.is_goal(current):    

                data = []

                while current in camefrom:
                    data.append(current)
                    current = camefrom[current]

                # append start, reverse order, sothat the start comes first
                data.append(self._start)
                data.reverse()
                
                print(self.__class__.__name__, 'suceeded after', iter)

                return data
            
            closed.add(current)

            for neighbour in self.get_neighbours(current):
                
                gscore_tentative = self.get_gscore(current) + self.get_cost(current, neighbour)
                
                # is the neighbor in the closed set? is the g-score greater than in previous tests?
                if neighbour in closed and gscore_tentative >= self.get_gscore(neighbour):     
                    
                    continue

                # is the gscore smaller than in previous tests?
                if  gscore_tentative < self.get_gscore(neighbour) or neighbour not in [i[1] for i in open]:

                    #add results to the dictionaries
                    camefrom[neighbour] = current
                    self.set_gscore(neighbour, gscore_tentative)
                    self.set_fscore(neighbour, gscore_tentative + self.get_heuristic(neighbour))
    
                    heappush(open, (self.get_fscore(neighbour), neighbour))
        
        print(self.__class__.__name__, 'failed after', iter)

        pass


class GridGraph(SearchGraph):

    def __init__(self, ndim: int = 3, directions: np.ndarray = None, heuristic: _Metric = None, cost: _Metric = None, **kwargs) -> None:
        
        self._ndim = ndim
        self._u    = np.zeros(self._ndim)
        self._v    = np.zeros(self._ndim)
   
        # allowed directions for getting potential neighbours
        # E.g. [0, 1] -> no neighbours in x-direction
        if directions is None:
            directions = np.ones(self._ndim).astype(int)
        else:
            directions = np.array(directions)
        
        self._directions = directions
        
        # neighbours
        self._neighbours = np.zeros((2*sum(directions > 0), self._ndim)).astype(directions.dtype)

        j = 0      
        for i in range(len(self._directions)):  
            if self._directions[i] > 0:
                self._neighbours[j*2  ,i] =  self._directions[i]
                self._neighbours[j*2+1,i] = -self._directions[i]
                j += 1

        if cost is None:
            self.cost = L1Metric()
        else:
            self.cost = cost

        if heuristic is None:
            self.heuristic = L1Metric()
        else:
            self.heuristic = heuristic

        super().__init__(**kwargs)

    def _get_potential_neighbours(self, u: Tuple) -> List[Tuple]:

        neighbours = u + (self._transform.R @ self._neighbours.T).T.astype(self._directions.dtype)

        return list(map(tuple, neighbours))

    def _calc_cost(self, u: Tuple, v: Tuple) -> float:
        
        self._u[:] = u
        self._v[:] = v

        return self.cost.eval(self._u, self._v)

    def _calc_heuristic(self, u: Tuple) -> float:
        
        self._u[:] = u
        self._v[:] = self._goal

        return self.heuristic.eval(self._u, self._v)

    def _calc_reachable(self, u: Tuple) -> bool:
        
        return True
    
    def _calc_traversable(self, u: Tuple, v: Tuple) -> bool:

        return True

    def is_goal(self, u: Tuple) -> bool:

        return u == self._goal

    def search(self, start: np.ndarray = np.zeros(3), goal: np.ndarray = np.zeros(3), transform: NdTransformMatrix = None) -> Path6:

        # coordinate frame used for getting potential neighbours
        if transform is None:
            transform = NdTransformMatrix(ndim = self._ndim)
        else:
            transform = NdTransformMatrix(transform)
        
        self._transform = transform

        data = super().search(tuple(start), tuple(goal))

        if data is not None:
            return Path6(data)
        else:
            pass

class MapGraph(GridGraph):

    def __init__(self, map: np.ndarray, directions: np.ndarray = None, **kwargs) -> None:
        
        self._map = map

        super().__init__(ndim = 3, directions = directions.astype(int), **kwargs)

    def _calc_reachable(self, u: Tuple) -> bool:
        
        self._u[:] = u

        if all(self._u < self._map.shape) and all(self._u >= 0):
            
            return self._map[u]
        
        return False


class TbPlatformPoseGraph(GridGraph):

    def __init__(self, goal_dist: float = 0, goal_skew: float = 0, directions: np.ndarray = None, feasiblity: FeasibilityContainer = None, **kwargs) -> None:
        
        self._goal_skew = goal_skew
        self._goal_dist = goal_dist

        if feasiblity is None:
            self.feasibility = FeasibilityContainer([TbWrenchFeasibility(10, 6)])
        else:
            self.feasibility = feasiblity
            
        super().__init__(ndim = 6, directions = directions, **kwargs) 
        
    def _get_potential_neighbours(self, u: Tuple) -> List[Tuple]:
        
        neighbours = np.round(u + (self._transform.R @ self._neighbours.T).T.astype(self._directions.dtype), 4)

        return list(map(tuple, neighbours))

    def _calc_reachable(self, u: Tuple) -> bool:
        
        self._tetherbot.platform.T_world = self._tetherbot.platform.T_world.compose(u)   
            
        return self.feasibility.eval(self._tetherbot)

    def is_goal(self, u: Tuple) -> bool:

        self._u[:] = u
        self._v[:] = self._goal

        return sqrt(sum((self._u[:3] - self._v[:3])**2)) <= self._goal_dist and sqrt(sum((self._u[3:] - self._v[3:])**2)) <= self._goal_skew

    def search(self, tetherbot: TbTetherbot, start: np.ndarray = None, goal: np.ndarray = np.zeros(6)) -> Path6:

        self._tetherbot = deepcopy(tetherbot)
        
        if start is None:
            start = self._tetherbot.platform.T_world.decompose()      

        # Coordinate frame for the search
        R = np.identity(6)
        R[:3,:3] = tbbasefit(self._tetherbot, output_format=0)[1]
        R[3:,3:] = basefit(np.vstack([start[3:],goal[3:]]), axis = 1)[1]
        # NOTE: a_world of inactive grippers are filtered with the tensioned property

        transform = NdTransformMatrix(start, R)

        path = super().search(start, goal, transform)

        if path is not None:
            path.replace(self._goal)
        
        return path
    

class TbPlatformAlignGraph(TbPlatformPoseGraph):

    def __init__(self, heuristic: _Metric = None, cost: _Metric = None, **kwargs) -> None:

        if heuristic is None:
            heuristic = TbAlignmentMetric()
        if cost is None:
            cost = ConstantMetric(0)
        
        super().__init__(goal_dist = None, heuristic = heuristic, cost = cost, **kwargs)
 
    def _calc_heuristic(self, u: Tuple) -> float:

        return self.heuristic.eval(u, self._goal1, self._goal2, self._tetherbot)

    def is_goal(self, u: Tuple) -> bool:
        
        self._tetherbot.platform.T_world = self._tetherbot.platform.T_world.compose(u)
        qs1 = self._tetherbot.platform.arm.ivk(TransformMatrix(self._goal1))
        qs2 = self._tetherbot.platform.arm.ivk(TransformMatrix(self._goal2))

        return self._tetherbot.platform.arm.valid(qs1) and self._tetherbot.platform.arm.valid(qs2) and ang3(self._tetherbot.platform.T_world.R[:,2], self._goal2[:3]-self._goal1[:3]) <= self._goal_skew

    def search(self, tetherbot: TbTetherbot, start: np.ndarray = None, goal: np.ndarray = np.zeros((2,6))) -> Path6:
        
        self._goal1 = goal[0] #primary goal (point for docking)
        self._goal2 = goal[1] #secondary goal (point for hovering)
       
        self._tetherbot = deepcopy(tetherbot)

        if start is None:
            start = self._tetherbot.platform.T_world.decompose()      
        
        # Coordinate frame for the search
        R = np.identity(6)
        R[:3,:3] = tbbasefit(self._tetherbot, output_format = 0)[1]
        R[3:,3:] = basefit(np.vstack((start[3:], self._goal1[3:])), axis = 1)[1] #R[3:,3:] = basefit(np.vstack([start[3:],goal[3:]]), axis = 1)[1]
        # NOTE: a_world of inactive grippers are filtered with the tensioned property

        transform = NdTransformMatrix(start, R)

        return super(TbPlatformPoseGraph, self).search(start, self._goal1, transform)


class TbArmPoseGraph(GridGraph):

    def __init__(self, goal_dist: float = 0, directions: np.ndarray = None, feasiblity: FeasibilityContainer = None, **kwargs) -> None:
        
        self._goal_dist = goal_dist

        if feasiblity is None:
            self.feasiblity = FeasibilityContainer([])
        else:
            self.feasiblity = feasiblity

        super().__init__(ndim = 3, directions = directions, **kwargs) 

    def _get_potential_neighbours(self, u: Tuple) -> List[Tuple]:

        neighbours = np.round(u + (self._transform.R @ self._neighbours.T).T.astype(self._directions.dtype), 4)

        return list(map(tuple, neighbours))
       
    def _calc_reachable(self, u: Tuple) -> bool:

        self._tetherbot.platform.arm.qs = self._tetherbot.platform.arm.ivk(TransformMatrix(u))

        return self.feasiblity.eval(self._tetherbot)

    def is_goal(self, u: Tuple) -> bool:

        self._u[:] = u
        self._v[:] = self._goal

        return sqrt(sum((self._u[:3] - self._v[:3])**2)) <= self._goal_dist

    def search(self, tetherbot: TbTetherbot, start: np.ndarray = None, goal: np.ndarray = None) -> Path6:
        
        if goal is None:
            goal = np.zeros(3)
        
        self._tetherbot = deepcopy(tetherbot)
        self._tetherbot._update_transforms()
        
        if start is None:
            start = self._tetherbot.platform.arm.links[-1].T_world.r
 
        # allows passing a goal via a transform matrix T with T.decompose() without adding a [:3] every time
        goal = goal[:3]

        # Best fit coordinate frame for the search (first base vector points to goal)
        transform = NdTransformMatrix(self._tetherbot.platform.arm.T_world.r, basefit(np.vstack([start, goal]), axis=1)[1])

        path = super().search(start, goal, transform)

        if path is not None:
            if path.length == 1:
                path.append(self._goal)
            else:
                path.replace(self._goal)

        return path
    

class TbGlobalGraph2(SearchGraph):

    def __init__(self, goal_dist: float = 0, platform2gripper: PlanPlatform2Gripper = None, platform2hold: PlanPlatform2Hold = None, workspace: TbWorkspace = None, heuristic: _Metric = None, cost: _Metric = None, **kwargs) -> None:
        
        self._goal_dist = goal_dist
        self._platform2hold = platform2hold
        self._platform2gripper = platform2gripper
        self._workspace = workspace
        
        self.feasiblity = self._workspace.feasiblity

        if heuristic is None:
            self.heuristic = StanceDisplacementMetric()
        else:
            self.heuristic = heuristic

        if cost is None:
            self.cost = ConstantMetric(0)
        else:
            self.cost = cost
        
        super().__init__(auto_clear = False, **kwargs)

    def get_neighbours(self, u: Tuple) -> List[Tuple]:

        for grip_idx, hold_idx in zip(range(self._tetherbot.k), u):
                if hold_idx == -1:
                    self._tetherbot.pick(grip_idx, True)
                else:
                    self._tetherbot.place(grip_idx, hold_idx, True)

        return super().get_neighbours(u)

    def _get_potential_neighbours(self, u: Tuple) -> List[Tuple]:
        
        # k-1 stance
        neighbours = []
        if -1 in u:
            grip_idx = u.index(-1)
            for hold_idx in self._tetherbot.filter_holds(grip_idx, self._C):
                neighbour = list(u)
                neighbour[grip_idx] = hold_idx
                if len(neighbour) == len(set(neighbour)):
                    neighbours.append(neighbour)
        # k stance
        else:
            for grip_idx in range(len(u)):
                neighbour = list(u)
                neighbour[grip_idx] = -1
                neighbours.append(neighbour)

        print(u)
        print('neigbhours ', neighbours)

        return list(map(tuple, neighbours))
    
    def _calc_heuristic(self, u: Tuple) -> float:

        return self.heuristic.eval(u, self._goal, self._tetherbot)

    def _calc_cost(self, u: Tuple, v: Tuple) -> float:

        return self.cost.eval(u, v, self._tetherbot)
    
    def _calc_reachable(self, u: Tuple) -> bool:
        # Note: Using u to place grippers not necessary as it was already done by get_neighbours

        # k-1 stance
        if -1 in u:
            grip_idx = u.index(-1)
            self._tetherbot.tension(grip_idx, False)
            reachable, self._reachable_pose = self._workspace.calculate(self._tetherbot)
            self._tetherbot.tension(grip_idx, True)
        # k stance
        else:
            # first check if there are any k-1 stances with stable poes
            # usually a stable pose in the k-1 stance is also stable for the k stance (not necessarly though, for example cable limits might not be met)
            for i in range(self._tetherbot.k):
                # create k-1 stance
                v    = list(u)
                v[i] = -1
                v = tuple(v)
                # check if k-1 stance is in the grpah
                reachable = False
                if self._graph.has_node(v):
                    # check if k-1 stance is feasible
                    if self.get_reachable(v):
                        # check if reachable pose of the k-1 stance is feasible for the k stance
                        self._reachable_pose = self.get_reachable_pose(v)
                        self._tetherbot.platform.T_local = TransformMatrix(self._reachable_pose)
                        reachable = self.feasiblity.eval(self._tetherbot)
                        if reachable:
                            break
                if not reachable:
                    reachable, self._reachable_pose = self._workspace.calculate(self._tetherbot)
        print(u, reachable)
        # Note: Reachable pose will be added by add_node
        return reachable > 0
    
    def add_node(self, u: Tuple) -> None:

        super().add_node(u)

        self.set_reachable_pose(u, self._reachable_pose)
    
    def _calc_traversable(self, u: Tuple, v: Tuple) -> bool:
        # Note: Using u to place grippers not necessary as it was already done by get_neighbours

        # k-1 stance
        if -1 in u:
            # place
            grip_idx = u.index(-1)
            hold_idx = v[grip_idx]
            self._tetherbot.platform.T_world = TransformMatrix(self.get_reachable_pose(u))
            self._tetherbot.tension(grip_idx, False)
            traversable = self._platform2hold.plan(self._tetherbot, hold_idx)[2]
            self._tetherbot.tension(grip_idx, True)
        # k stance
        elif -1 in v:     
            # pick
            grip_idx = v.index(-1)
            self._tetherbot.platform.T_world = TransformMatrix(self.get_reachable_pose(v)) #u
            self._tetherbot.tension(grip_idx, False)
            traversable = self._platform2gripper.plan(self._tetherbot, grip_idx)[2]
            self._tetherbot.tension(grip_idx, True)

        return traversable is not None
    
    def set_reachable_pose(self, u: Tuple, value: np.ndarray) -> None:

        self._graph.nodes[u]['reachable_pose'] = value

    def get_reachable_pose(self, u: Tuple) -> np.ndarray:

        return self._graph.nodes[u]['reachable_pose']
    
    def is_goal(self, u: Tuple) -> bool:

        if -1 in u:
            return False
        else:
            return self.get_heuristic(u) <= self._goal_dist

    def search(self, tetherbot: TbTetherbot, start: Tuple, goal: Tuple) -> ClimbPath: 
        
        self._tetherbot = deepcopy(tetherbot)
        self._k         = self._tetherbot.k
        self._C         = self._tetherbot.C_world
        self._u         = np.empty(self._k)
        self._v         = np.empty(self._k)

        self._tetherbot.remove_all_geometries()
        self._tetherbot.toggle_fast_mode(True)
        
        data = super().search(start, goal)

        if data is not None: 
            return ClimbPath(data)
        else:
            pass