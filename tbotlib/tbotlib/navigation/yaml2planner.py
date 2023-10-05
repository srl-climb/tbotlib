from __future__ import annotations
from .Planner import GlobalPlanner, PlanArm2Pose, PlanPickAndPlace2, PlanPlatform2Gripper, PlanPlatform2Configuration, PlanPlatform2Hold, PlanPlatform2Pose, FastPlanPickAndPlace, FastPlanPlatform2Configuration
from .Smoother import BsplineSmoother
from .Profile import ProfileQArm, ProfileQPlatform, FastProfile
from .Graph import TbArmPoseGraph, TbGlobalGraph, TbPlatformAlignGraph, TbPlatformPoseGraph
from .Workspace import TbWorkspace
import yaml

def yaml2planner(file: str):

    with open(file, "r") as stream:
        data = yaml.safe_load(stream)
        
    simulation_dt = data['simulation']['dt']

    # smoothing algorithms
    platform_smoother = BsplineSmoother(data['platform']['smoothing']['ds'])
    arm_smoother = BsplineSmoother(data['arm']['smoothing']['ds'])

    # trajectory generators
    platform_profiler = ProfileQPlatform(platform_smoother, 
                                            simulation_dt, 
                                            t_t = None, 
                                            v_t = data['platform']['profiler']['v_t'],
                                            qlim = data['platform']['profiler']['q_lim'])
    arm_profiler = ProfileQArm(arm_smoother, 
                                simulation_dt, 
                                t_t = None, 
                                v_t = data['arm']['profiler']['v_t'],
                                qlim = data['arm']['profiler']['q_lim'])
    
    # local planners
    platform2pose = PlanPlatform2Pose(
        graph = TbPlatformPoseGraph(goal_dist = data['platform']['planner']['platform_to_pose']['graph']['goal_dist'], 
                                    goal_skew = data['platform']['planner']['platform_to_pose']['graph']['goal_skew'], 
                                    directions = data['platform']['planner']['platform_to_pose']['graph']['directions'],
                                    iter_max = data['platform']['planner']['platform_to_pose']['graph']['iter_max']),
        profiler = platform_profiler)
    platform2configuration = PlanPlatform2Configuration(
        graph = TbPlatformPoseGraph(goal_dist = data['platform']['planner']['platform_to_configuration']['graph']['goal_dist'], 
                                    goal_skew = data['platform']['planner']['platform_to_configuration']['graph']['goal_skew'], 
                                    directions = data['platform']['planner']['platform_to_configuration']['graph']['directions'],
                                    iter_max = data['platform']['planner']['platform_to_configuration']['graph']['iter_max']),
        profiler = platform_profiler,
        workspace = TbWorkspace(padding = data['platform']['planner']['platform_to_configuration']['workspace']['padding'], 
                                scale = data['platform']['planner']['platform_to_configuration']['workspace']['scale'], 
                                mode = data['platform']['planner']['platform_to_configuration']['workspace']['mode'],
                                mode_2d = data['platform']['planner']['platform_to_configuration']['workspace']['mode_2d']))
    platform2gripper = PlanPlatform2Gripper(
        graph = TbPlatformAlignGraph(goal_skew = data['platform']['planner']['platform_to_gripper']['graph']['goal_skew'], 
                                        directions = data['platform']['planner']['platform_to_gripper']['graph']['directions'], 
                                        iter_max = data['platform']['planner']['platform_to_gripper']['graph']['iter_max']),
                                        profiler = platform_profiler)
    platform2hold = PlanPlatform2Hold(
        graph = TbPlatformAlignGraph(goal_skew = data['platform']['planner']['platform_to_hold']['graph']['goal_skew'], 
                                        directions = data['platform']['planner']['platform_to_hold']['graph']['directions'], 
                                        iter_max = data['platform']['planner']['platform_to_hold']['graph']['iter_max']),
        profiler = platform_profiler)
    arm2pose = PlanArm2Pose(
        graph = TbArmPoseGraph(goal_dist = data['arm']['planner']['arm_to_pose']['graph']['goal_dist'],
                                directions = data['arm']['planner']['arm_to_pose']['graph']['directions'], 
                                iter_max = data['arm']['planner']['arm_to_pose']['graph']['iter_max']),
        profiler = arm_profiler)
    local_planner = PlanPickAndPlace2(
        platform2configuration = platform2configuration,
        platform2gripper = platform2gripper, 
        platform2hold = platform2hold, 
        arm2pose = arm2pose)
    
    # global planner
    platform2configuration = FastPlanPlatform2Configuration(
        workspace = TbWorkspace(padding = data['global']['planner']['fast_platform_to_configuration']['workspace']['padding'], 
                                scale = data['global']['planner']['fast_platform_to_configuration']['workspace']['scale'], 
                                mode = data['global']['planner']['fast_platform_to_configuration']['workspace']['mode'],
                                mode_2d = data['global']['planner']['fast_platform_to_configuration']['workspace']['mode_2d']))
    platform2gripper = PlanPlatform2Gripper(
        graph = TbPlatformAlignGraph(goal_skew = data['global']['planner']['platform_to_gripper']['graph']['goal_skew'], 
                                        directions = data['global']['planner']['platform_to_gripper']['graph']['directions'], 
                                        iter_max = data['global']['planner']['platform_to_gripper']['graph']['iter_max']),
        profiler = FastProfile())
    platform2hold = PlanPlatform2Hold(
        graph = TbPlatformAlignGraph(goal_skew = data['global']['planner']['platform_to_hold']['graph']['goal_skew'], 
                                        directions = data['global']['planner']['platform_to_hold']['graph']['directions'], 
                                        iter_max = data['global']['planner']['platform_to_hold']['graph']['iter_max']),
        profiler = FastProfile())
    fast_local_planner = FastPlanPickAndPlace(
        platform2configuration = platform2configuration,
        platform2gripper = platform2gripper,
        platform2hold = platform2hold)
    global_planner = GlobalPlanner(
        graph = TbGlobalGraph(goal_dist = data['global']['graph']['goal_dist'], 
                                planner = fast_local_planner, 
                                workspace = TbWorkspace(padding = data['global']['graph']['workspace']['padding'],
                                                        scale = data['global']['graph']['workspace']['scale'],
                                                        mode = data['global']['graph']['workspace']['mode'],
                                                        mode_2d = data['global']['graph']['workspace']['mode_2d']),
                                iter_max = data['global']['graph']['iter_max']),
        localplanner = local_planner)
        
    return simulation_dt, platform2pose, platform2configuration, arm2pose, local_planner, global_planner 