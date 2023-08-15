from __future__ import annotations

import numpy as np

from ..tools import hyperRectangle, Ring
from .TbPlatform import TbPlatform
from .TbArm import TbRPPArm
from .TbLink import TbRevoluteLink, TbPrismaticLink
from .TbGeometry import TbCylinder, TbSphere, TbAlphashape, TbTethergeometry
from .TbTetherbot import TbTetherbot
from .TbGripper import TbGripper
from .TbPoint import TbAnchorPoint, TbCamera, TbDepthsensor, TbMarker
from .TbTether import TbTether
from .TbHold import TbHold
from .TbWall import TbWall
from math import pi

def tbExample() -> TbTetherbot:

    # create telescopic arm
    # first link
    geometries = [TbCylinder(radius=0.050, height=0.05, T_local=[0,0,0,90,0,0]), 
                TbCylinder(radius=0.016, height=0.30, T_local=[0,0,0.15,0,0,0])]   
    link_1 = TbRevoluteLink(q0=0, alpha=-pi/2, a=0, d=0.095, qlim=[-pi, pi], geometries=geometries)
    
    # second link
    geometries = [TbCylinder(radius=0.014, height=1.200, T_local=[0,0.616,0,90,0,0]), 
                TbCylinder(radius=0.014, height=0.016, T_local=[0,0.007,0,90,0,0]),
                TbCylinder(radius=0.016, height=0.050, T_local=[0,0,0,0,0,0])] 
    link_2 = TbPrismaticLink(phi=0, alpha=-pi/2, a=0, q0=0.314, qlim=[0.314,1.414], geometries=geometries)
    
    # third link
    geometries = [TbCylinder(radius=0.014, height=0.2, T_local=[0,0,-0.1,0,0,0])]   
    link_3 = TbPrismaticLink(phi=0, alpha=0, a=0, q0=0.04, qlim=[0,0.3], geometries=geometries)
    
    # arm
    arm = TbRPPArm(T_local=[0,0,0,0,0,0], links=[link_1, link_2, link_3])

    # create grippers
    grippers = []
    for _ in range(5):
        geometries = [TbCylinder(T_local=[0,0,0.015], radius=0.015, height=0.03), TbSphere(T_local=[0,0,0.03], radius=0.02)]
        marker = TbMarker()
        gripper = TbGripper.create(hoverpoint=[0,0,0.1], grippoint=[0,0,0], anchorpoint=[0,0,0.03], dockpoint=[0,0,0.05], geometries=geometries, marker=marker)

        grippers.append(gripper)

    # create tethers
    tethers = []
    for _ in range(10):
        geometries = [TbTethergeometry(radius=0.008)]
        tether = TbTether.create(f_min=0, f_max=1000, geometries=geometries)

        tethers.append(tether)

    # create platform
    # anchorpoints
    points = np.array([[ 0.2, 0,    0.05],
                       [ 0.2, 0,   -0.05],
                       [ 0.2, 0.15, 0.05],
                       [ 0.2, 0.15,-0.05],
                       [-0.2, 0.15, 0.05],
                       [-0.2, 0.15,-0.05],
                       [-0.2,-0.15, 0.05],
                       [-0.2,-0.15,-0.05],
                       [ 0.2,-0.15, 0.05],
                       [ 0.2,-0.15,-0.05]])
    anchorpoints = [TbAnchorPoint(T_local=point) for point in points]

    # cameras
    cameras = []
    for _ in range(5):
        camera = TbCamera()

        cameras.append(camera)

    #depthsensor
    depthsensors = [TbDepthsensor()]
    
    #platform
    geometries = [TbAlphashape(points=points, alpha=1), TbCylinder(radius=0.05, height=0.02, T_local=[0,0,0.06])]
    platform = TbPlatform(T_local = [0,0,2.15], geometries=geometries, arm=arm, anchorpoints=anchorpoints, cameras=cameras, depthsensors=depthsensors)

    # create wall
    points = np.array([[ 0.5, 0,   2.15], 
                       [ 0.5, 0.5, 2.15], 
                       [ 0.5,-0.5, 2.15], 
                       [-0.5, 0.5, 2.15],
                       [-0.5,-0.5, 2.15], 
                       [ 0.8, 0,   2.15], 
                       [ 0.8, 0.5, 2.15], 
                       [ 0.8,-0.5, 2.15], 
                       [-0.2, 0.5, 2.15],
                       [-0.2,-0.5, 2.15]])
    
    # holds
    holds = []
    for point in points:
        geometries  = [TbCylinder(radius=0.05, height=0.03)]
        hold = TbHold.create(T_local=point, hoverpoint=[0,0,0.05], grippoint=[0,0,0], geometries=geometries)

        holds.append(hold)

    # wall
    wall = TbWall(holds=holds)


    # create tetherbot
    wrench = hyperRectangle(np.array([5,5,5,0.5,0.5,0.5]), np.array([-5,-5,-5,-0.5,-0.5,-0.5]))
    mapping = [[0,0],[0,1],[1,2],[1,3],[3,4],[3,5],[4,6],[4,7],[2,8],[2,9]]
    aorder = Ring([0,1,3,4,2]) #indices of the grippers counter clockwise

    tetherbot = TbTetherbot(platform=platform, grippers=grippers, tethers=tethers, wall=wall, W=wrench, mapping=mapping, aorder=aorder)
    
    return tetherbot