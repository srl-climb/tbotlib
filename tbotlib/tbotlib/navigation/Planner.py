from __future__     import annotations
from ..matrices     import TransformMatrix
from ..tetherbot    import TbTetherbot
from .Workspace     import *
from .Graph         import *
from .Command       import *
from .Profile       import *
from .Smoother      import *
from typing         import Type, Tuple
from abc            import ABC, abstractmethod
import numpy as np

class AbstractPlanner(ABC):

    def __init__(self) -> None:
        
        pass

    @abstractmethod
    def plan(self) -> None:

        pass

class LocalPlanner(AbstractPlanner):

    def __init__(self, graph: Type[SearchGraph], profiler: Type[AbstractProfile]) -> None:
        
        self._graph     = graph
        self._profiler  = profiler

    def plan(self, tetherbot: TbTetherbot, goal: np.ndarray) -> tuple[TbTetherbot, Profile, bool]:
        
        path     = self._graph.search(tetherbot, start = None, goal = goal)
        profile  = self._profiler.calculate(path, tetherbot=tetherbot)
        exitflag = profile is not None

        return tetherbot, profile, exitflag

class PlanPlatform2Pose(LocalPlanner):

    def __init__(self, graph: TbPlatformPoseGraph = None, profiler: Type[AbstractProfile] = None) -> None: 

        if graph is None:
            graph = TbPlatformPoseGraph(goal_dist=0.025, goal_skew=1, directions=[0.02,0.02,0.02,1,0,0])
        if profiler is None:
            profiler = SlicedProfile6(a_t=[0.05,1], d_t=[0.05,1], v_t=[0.75,0.5], dt=0.0167, smoother = BsplineSmoother(0.001))

        super().__init__(graph, profiler)

    def plan(self, tetherbot: TbTetherbot, pose: TransformMatrix, commands: CommandList = None) -> Tuple[TbTetherbot, CommandList, Profile]:

        tetherbot, profile, exitflag = super().plan(tetherbot, pose.decompose())
        
        if commands is not None and exitflag:
            commands.append(CommandMovePlatform(profile))
        
        if exitflag:
            tetherbot.platform.T_world = profile.poses[-1]

        return tetherbot, commands, profile


class PlanPlatform2Gripper(LocalPlanner):

    def __init__(self, graph: TbPlatformAlignGraph = None, profiler: Type[AbstractProfile] = None) -> None: 

        if graph is None:
            graph = TbPlatformAlignGraph(goal_skew=1, directions=[0.01,0.01,0.01,1,0,0])
        if profiler is None:
            profiler = SlicedProfile6(a_t=[0.05,1], d_t=[0.05,1], v_t=[0.75,0.5], dt=0.0167, smoother = BsplineSmoother(0.001))

        super().__init__(graph, profiler)

    def plan(self, tetherbot: TbTetherbot, grip_idx: int, commands: CommandList = None) -> Tuple[TbTetherbot, CommandList, Profile]:

        goal = np.vstack((tetherbot.grippers[grip_idx].dockpoint.T_world.decompose(),
                          tetherbot.grippers[grip_idx].hoverpoint.T_world.decompose()))

        tetherbot, profile, exitflag = super().plan(tetherbot, goal)
        
        if commands is not None and exitflag:
            commands.append(CommandMovePlatform(profile))

        if exitflag:
            tetherbot.platform.T_world = profile.poses[-1]
            
        return tetherbot, commands, profile


class PlanPlatform2Hold(PlanPlatform2Gripper):

    def plan(self, tetherbot: TbTetherbot, hold_idx: int, commands: CommandList = None) -> Tuple[TbTetherbot, CommandList, Profile]:

        # offset because of the gripper docked to the endeffector of the arm
        offset = tetherbot.grippers[0].dockpoint.r_local - tetherbot.grippers[0].grippoint.r_local

        goal = np.zeros((2,6))
        
        goal[:,:3] = np.vstack((tetherbot.wall.holds[hold_idx].grippoint.r_world + offset,
                                tetherbot.wall.holds[hold_idx].hoverpoint.r_world + offset))

        tetherbot, profile, exitflag = super(PlanPlatform2Gripper, self).plan(tetherbot, goal)

        if commands is not None and exitflag:
            commands.append(CommandMovePlatform(profile))

        if exitflag:
            tetherbot.platform.T_world = profile.poses[-1]

        return tetherbot, commands, profile


class PlanPlatform2Configuration(PlanPlatform2Pose):

    def __init__(self, workspace: TbWorkspace = None, **kwargs) -> None: 

        if workspace is None:
            workspace = TbWorkspace(padding=[-0.1,-0.1,0,-180,-180,-135], scale=[0.1,0.1,0.1,45,45,45], mode = 'max')

        self._workspace = workspace

        super().__init__(**kwargs)

    def plan(self, tetherbot: TbTetherbot, grip_idx: int, commands: CommandList = None) -> Tuple[TbTetherbot, CommandList, Profile]:
      
        # find pose with the highest stability
        tetherbot.tension(grip_idx, False)
        stability, goal = self._workspace.calculate(tetherbot)
        tetherbot.tension(grip_idx, True)

        if stability > 0:
            tetherbot, profile, exitflag = super(PlanPlatform2Pose, self).plan(tetherbot, goal)
        else:
            profile  = None
            exitflag = False
        
        if commands is not None and exitflag:
            commands.append(CommandMovePlatform(profile))

        if exitflag:
            tetherbot.platform.T_world = profile.poses[-1]
        
        return tetherbot, commands, profile


class FastPlanPlatform2Configuration(PlanPlatform2Configuration):

    def plan(self, tetherbot: TbTetherbot, grip_idx: int) -> tuple[TbTetherbot, None, bool]:

        # find pose with the highest stability
        tetherbot.tension(grip_idx, False)
        stability, goal = self._workspace.calculate(tetherbot)
        tetherbot.tension(grip_idx, True)

        if stability > 0:
            exitflag = True
            tetherbot.platform.T_world = TransformMatrix(goal)
        else: 
            exitflag = False

        return tetherbot, None, exitflag


class PlanArm2Pose(LocalPlanner):

    def __init__(self, graph: TbArmPoseGraph = None, profiler: Type[AbstractProfile] = None, **kwargs) -> None:

        if graph is None:
            graph = TbArmPoseGraph(goal_dist=0.05, directions=[0.025,0.025,0.025])
        
        if profiler is None:
            profiler = Profile3(a_t=0.05, d_t=0.05, v_t=0.75, dt=0.0167, smoother = BsplineSmoother(0.001))

        super().__init__(graph, profiler, **kwargs)

    def plan(self, tetherbot: TbTetherbot, pose: TransformMatrix, commands: CommandList = None) -> Tuple[TbTetherbot, CommandList, Profile]:
        
        tetherbot, profile, exitflag = super().plan(tetherbot, pose.decompose())

        if commands is not None and exitflag:
            commands.append(CommandMoveArm(profile))

        if exitflag:
            tetherbot.platform.arm.qs = tetherbot.platform.arm.ivk(profile.poses[-1])
            
        return tetherbot, commands, profile


class PlanPickAndPlace(AbstractPlanner):

    def __init__(self, platform2configuration: PlanPlatform2Configuration = None,
                 platform2gripper: PlanPlatform2Gripper = None, platform2hold: PlanPlatform2Hold = None, arm2pose: PlanArm2Pose = None) -> None:
        
        if platform2configuration is None:
            platform2configuration = PlanPlatform2Configuration()

        if platform2gripper is None:
            platform2gripper = PlanPlatform2Gripper()

        if platform2hold is None:
            platform2hold = PlanPlatform2Hold()

        if arm2pose is None:
            arm2pose = PlanArm2Pose()
        
        self._platform2configuration = platform2configuration
        self._platform2gripper       = platform2gripper
        self._platform2hold          = platform2hold
        self._arm2pose               = arm2pose
     
    def plan(self, tetherbot: TbTetherbot, grip_idx: int, hold_idx: int, commands: CommandList = None) -> Tuple[TbTetherbot, CommandList, bool]:

        # move platform into stable position
        tetherbot, commands, exitflag = self._platform2configuration.plan(tetherbot, grip_idx, commands)
        
        if exitflag is None: return tetherbot, commands, False

        # move platform close to gripper
        tetherbot, commands, exitflag = self._platform2gripper.plan(tetherbot, grip_idx, commands)
        
        if exitflag is None: return tetherbot, commands, False
        
        # move arm to gripper hoverpoint
        tetherbot, commands, exitflag = self._arm2pose.plan(tetherbot, tetherbot.grippers[grip_idx].hoverpoint.T_world, commands)

        if exitflag is None: return tetherbot, commands, False

        # untension the tethers of the gripper
        tetherbot.tension(grip_idx, False)

        if commands is not None:
            commands.append(CommandTensionTethers(grip_idx, False))

        # move arm to gripper dockpoint
        tetherbot, commands, exitflag = self._arm2pose.plan(tetherbot, tetherbot.grippers[grip_idx].dockpoint.T_world, commands)

        if exitflag is None: return tetherbot, commands, False

        # pick gripper
        tetherbot.pick(grip_idx)

        if commands is not None:
            commands.append(CommandPickGripper(grip_idx))
        
        # move arm with gripper to hoverpoint
        tetherbot, commands, exitflag = self._arm2pose.plan(tetherbot, tetherbot.grippers[grip_idx].hoverpoint.T_world, commands)

        if exitflag is None: return tetherbot, commands, False

        # move platform close to hold
        tetherbot, commands, exitflag = self._platform2hold.plan(tetherbot, hold_idx, commands)

        if exitflag is None: return tetherbot, commands, False

        # move gripper with arm to hoverpoint
        tetherbot, commands, exitflag = self._arm2pose.plan(tetherbot, 
            TransformMatrix(tetherbot.wall.holds[hold_idx].hoverpoint.r_world + tetherbot.grippers[grip_idx].dockpoint.r_local - tetherbot.grippers[grip_idx].grippoint.r_local), 
            commands)

        if exitflag is None: return tetherbot, commands, False
        
        # move gripper with arm to grippoint
        tetherbot, commands, exitflag = self._arm2pose.plan(tetherbot,
            TransformMatrix(tetherbot.wall.holds[hold_idx].grippoint.r_world + tetherbot.grippers[grip_idx].dockpoint.r_local - tetherbot.grippers[grip_idx].grippoint.r_local), 
            commands)
        
        if exitflag is None: return tetherbot, commands, False

        # place gripper
        tetherbot.place(grip_idx, hold_idx, correct_pose = True)

        if commands is not None:
            commands.append(CommandPlaceGripper(grip_idx, hold_idx))

        # move arm to gripper hoverpoint
        tetherbot, commands, exitflag = self._arm2pose.plan(tetherbot, tetherbot.grippers[grip_idx].hoverpoint.T_world, commands)

        if exitflag is None: return tetherbot, commands, False

        # tension the tethers of the gripper
        tetherbot.tension(grip_idx, True)

        if commands is not None:
            commands.append(CommandTensionTethers(grip_idx, True))

        return tetherbot, commands, True

class PlanPickAndPlace2(PlanPickAndPlace):

    def plan(self, tetherbot: TbTetherbot, grip_idx: int, hold_idx: int, commands: CommandList = None, start_state: int = 0, goal_state: int = 10) -> Tuple[TbTetherbot, CommandList, bool]:
        # States
        # 0:  move platform into stable position
        # 1:  move platform close to gripper
        # 2:  move arm to gripper hover point
        # 3:  move arm to gripper dock point
        # 4:  pick gripper
        # 5:  move arm with gripper to hover point
        # 6:  move platform close to hold
        # 7:  move gripper with arm to hoverpoint
        # 8:  move gripper with arm to grippoint
        # 9:  place gripper
        # 10: move arm to gripper hoverpoint

        print('Plan pick and place of Gripper (' + str(grip_idx) + ') to Hold (' + str(hold_idx) + ')')

        next_state = start_state
        while True:
            current_state = next_state

            if current_state == 0:
                # move platform into stable position
                tetherbot, commands, exitflag = self._platform2configuration.plan(tetherbot, grip_idx, commands)
                next_state = 1
            elif current_state == 1:
                # move platform close to gripper
                tetherbot, commands, exitflag = self._platform2gripper.plan(tetherbot, grip_idx, commands)
                next_state = 2
                # NOTE: Actual tensioning happens during CommandPickGripper and CommandPlaceGripper,
                # we only (un)tension the gripper here so that the path planner keeps the platform at a pose which is stable without the gripper to be picked/placed
            elif current_state == 2:
                # move arm to gripper hoverpoint
                tetherbot, commands, exitflag = self._arm2pose.plan(tetherbot, tetherbot.grippers[grip_idx].hoverpoint.T_world, commands)
                next_state = 3
            elif current_state == 3:
                # untension the tethers of the gripper
                tetherbot.tension(grip_idx, False)
                if commands is not None:
                    commands.append(CommandTensionTethers(grip_idx, False))
                # move arm to gripper dockpoint
                tetherbot, commands, exitflag = self._arm2pose.plan(tetherbot, tetherbot.grippers[grip_idx].dockpoint.T_world, commands)
                next_state = 4
            elif current_state == 4:
                # pick gripper
                tetherbot.pick(grip_idx)
                if commands is not None:
                    commands.append(CommandPickGripper(grip_idx))
                next_state = 5
            elif current_state == 5:
                # move arm with gripper to hoverpoint
                tetherbot, commands, exitflag = self._arm2pose.plan(tetherbot, tetherbot.grippers[grip_idx].hoverpoint.T_world, commands)
                next_state = 6
            elif current_state == 6:
                # move platform close to hold
                tetherbot, commands, exitflag = self._platform2hold.plan(tetherbot, hold_idx, commands)
                next_state = 7
            elif current_state == 7:
                # move gripper with arm to hoverpoint
                tetherbot, commands, exitflag = self._arm2pose.plan(
                    tetherbot, 
                    TransformMatrix(tetherbot.wall.holds[hold_idx].hoverpoint.r_world + tetherbot.grippers[grip_idx].dockpoint.r_local - tetherbot.grippers[grip_idx].grippoint.r_local), 
                    commands)
                next_state = 8
            elif current_state == 8:
                # move gripper with arm to grippoint
                tetherbot, commands, exitflag = self._arm2pose.plan(
                    tetherbot,
                    TransformMatrix(tetherbot.wall.holds[hold_idx].grippoint.r_world + tetherbot.grippers[grip_idx].dockpoint.r_local - tetherbot.grippers[grip_idx].grippoint.r_local), 
                    commands)
                next_state = 9
            elif current_state == 9:
                # place gripper
                tetherbot.place(grip_idx, hold_idx, correct_pose = True)
                if commands is not None:
                    commands.append(CommandPlaceGripper(grip_idx, hold_idx))
                next_state = 10
            elif current_state == 10:
                # move arm to gripper hoverpoint
                tetherbot, commands, exitflag = self._arm2pose.plan(tetherbot, tetherbot.grippers[grip_idx].hoverpoint.T_world, commands)
                # tension the tethers of the gripper
                tetherbot.tension(grip_idx, True)
                if commands is not None:
                    commands.append(CommandTensionTethers(grip_idx, True))
                next_state = 0

            if exitflag is None: 
                return tetherbot, commands, False
            if current_state == goal_state:
                return tetherbot, commands, True    

class FastPlanPickAndPlace(AbstractPlanner):

    def __init__(self, platform2configuration: FastPlanPlatform2Configuration = None,
                 platform2gripper: PlanPlatform2Gripper = None, platform2hold: PlanPlatform2Hold = None) -> None:

        if platform2configuration is None:
            platform2configuration = FastPlanPlatform2Configuration(workspace = TbWorkspace(padding=[-0.1,-0.1,0,-180,-180,-135], scale=[0.1,0.1,0.1,45,45,45], mode = 'max')) #TbWorkspace(padding=[-0.1,-0.1,0,-180,-180,-135], scale=[0.1,0.1,0.1,45,45,45], mode = 'max')
                                           
        if platform2gripper is None:
            platform2gripper = PlanPlatform2Gripper()

        if platform2hold is None:
            platform2hold = PlanPlatform2Hold()

        self._platform2configuration = platform2configuration
        self._platform2gripper       = platform2gripper
        self._platform2hold          = platform2hold

    def plan(self, tetherbot: TbTetherbot, grip_idx: int, hold_idx: int) -> tuple[TbTetherbot, None, bool]:
        
        # move platform into stable position
        tetherbot, _, exitflag = self._platform2configuration.plan(tetherbot, grip_idx)
        
        if exitflag is False: return tetherbot, None, False

        # untension the tethers of the gripper
        tetherbot.tension(grip_idx, False)

        # move platform close to gripper
        tetherbot, _, exitflag = self._platform2gripper.plan(tetherbot, grip_idx)

        if exitflag is None: return tetherbot, None, False

        # SKIPPED: move arm to gripper hoverpoint, move arm to gripper dockpoint, pick gripper, move arm with gripper to hoverpoint

        # move platform close to hold
        tetherbot, _, exitflag = self._platform2hold.plan(tetherbot, hold_idx)
 
        if exitflag is None: return tetherbot, None, False

        # SKIPPED: move gripper with arm to hoverpoint, move gripper with arm to grippoint, place gripper

        # tension the tethers of the gripper
        tetherbot.tension(grip_idx, True)

        # SKIPPED: move arm to gripper hoverpoint

        return tetherbot, None, True


class GlobalPlanner(AbstractPlanner):

    def __init__(self, graph: TbGlobalGraph = None, localplanner: PlanPickAndPlace = None) -> None:
        
        if graph is None:
            graph = TbGlobalGraph(goal_dist=0.001, planner=FastPlanPickAndPlace(), workspace=TbWorkspace(padding=[-0.1,-0.1,0,-180,-180,-135], scale=[0.3,0.3,0.1,45,45,45], mode = 'first'))

        if localplanner is None:
            localplanner = PlanPickAndPlace()

        self._graph        = graph
        self._localplanner = localplanner

    def plan(self, tetherbot: TbTetherbot, start: np.ndarray, goal: np.ndarray, commands: CommandList = None) -> Tuple[TbTetherbot, CommandList, bool]:

        print()
        print('global path planning...')

        path = self._graph.search(tetherbot, tuple(start), tuple(goal))

        if path is not None:
            print('...successful!')
            print(path.coordinates)
            exitflag = True
        else:
            print('...failed.')
            exitflag = False

        if commands is not None and exitflag:

            for grip_idx, hold_idx in zip(path.grip_idc, path.hold_idc):
                tetherbot, commands, exitflag = self._localplanner.plan(tetherbot, grip_idx, hold_idx, commands)
    
                if exitflag == False:
                    break

        return tetherbot, commands, exitflag




