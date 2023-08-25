from .TbTetherbot import TbTetherbot
from .TbObject import TbObject
from .TbGeometry import TbBox, TbCylinder, TbSphere, TbAlphashape, TbTrianglemesh
from .TbLink import TbLink
from .TbGripper import TbGripper
from .TbPlatform import TbPlatform
from .TbPoint import TbPoint
from math import radians


def _create_base(basename: str, prefix: str) -> str:

    return ' <link name="'+ prefix + basename + '"/> \n\n'

def _create_link(obj: TbObject, filepath: str, stlpath: str, pointradius: float, prefix: str):

    urdf =  ' <link name="'+ prefix + obj.name + '"> \n'

    if isinstance(obj, TbBox):
        urdf += '  <visual> \n' \
                '   <geometry> \n' \
                '    <box size="' + str(obj.dimension[0]) + str(obj.dimension[1]) + str(obj.dimension[2]) + '"/> \n' \
                '   </geometry> \n' \
                '   <material name="Cyan"> \n' \
                '    <color rgba="0 1.0 1.0 1.0"/> \n' \
                '   </material> \n' \
                '  </visual> \n' 
    
    elif isinstance(obj, TbSphere):
        urdf += '  <visual> \n' \
                '   <geometry> \n' \
                '    <sphere radius="' + str(obj.radius) + '"/> \n' \
                '   </geometry> \n' \
                '   <material name="Cyan"> \n' \
                '    <color rgba="0 1.0 1.0 1.0"/> \n' \
                '   </material> \n' \
                '  </visual> \n' 
                
    elif isinstance(obj, TbPoint):
        urdf += '  <visual> \n' \
                '   <geometry> \n' \
                '    <sphere radius="' + str(pointradius) + '"/> \n' \
                '   </geometry> \n' \
                '   <material name="Cyan"> \n' \
                '    <color rgba="0 1.0 1.0 1.0"/> \n' \
                '   </material> \n' \
                '  </visual> \n' 
                
    elif isinstance(obj, TbCylinder):
        urdf += '  <visual> \n' \
                '   <geometry> \n' \
                '    <cylinder radius="' + str(obj.radius) + '" length="' + str(obj.height) + '"/> \n' \
                '   </geometry> \n' \
                '   <material name="Cyan"> \n' \
                '    <color rgba="0 1.0 1.0 1.0"/> \n' \
                '   </material> \n' \
                '  </visual> \n' 
        
    elif isinstance(obj, (TbAlphashape, TbTrianglemesh)):
        urdf += '  <visual> \n' \
                '   <geometry> \n' \
                '    <mesh filename="' + stlpath + '/' + obj.name + '.stl" scale="1 1 1"/> \n' \
                '   </geometry> \n' \
                '   <material name="Cyan"> \n' \
                '    <color rgba="0 1.0 1.0 1.0"/> \n' \
                '   </material> \n' \
                '  </visual> \n' 
        print(filepath + '/'+ prefix + obj.name + '.stl')
        obj.save_as_trianglemesh(filepath + '/'+ prefix + obj.name + '.stl')
                
    urdf += ' </link> \n\n'

    return urdf

def _create_joint(obj: TbObject, prefix: str):

    urdf =  ' <joint name="'+ prefix + obj.parent.name + '_to_'+ prefix + obj.name + '" type="'
    
    transform = obj.T_local.decompose()
    transform[3] = radians(transform[3])
    transform[4] = radians(transform[4])
    transform[5] = radians(transform[5])
    
    if  isinstance(obj, (TbGripper, TbPlatform, TbLink)):
        urdf += 'floating"> \n' 
    else:
        urdf += 'fixed"> \n' 


    urdf += '  <parent link="'+ prefix + obj.parent.name + '"/> \n' \
            '  <child link="'+ prefix + obj.name + '"/> \n' \
            '  <origin xyz="' + str(transform[0]) + ' ' + str(transform[1]) + ' ' + str(transform[2]) + '" rpy="' + str(transform[3]) + ' ' + str(transform[4]) + ' ' + str(transform[5]) + '"/> \n' \
            ' </joint> \n\n'

    return urdf

def tb2urdf(tetherbot: TbTetherbot, filepath: str, stlpath: str, prefix: str = '') -> str:

    # remove the tethers, we do not need them in the urdf file
    for tether in tetherbot.tethers:
        tether.anchorpoints[0]._remove_child(tether)
        tether.anchorpoints[1]._remove_child(tether)

    # while in the TbTetherbot, the grippers have different parents, in the urdf file we want the parent to be the map
    for gripper in tetherbot.grippers:
        gripper.parent = tetherbot

    urdf = ''
    urdf += '<?xml version="1.0"?> \n\n'
    urdf += '<robot name="' + tetherbot.name + '"> \n\n'
    urdf += _create_base('map', prefix)

    tetherbot.name = 'map'

    for child in tetherbot.get_all_children():
        urdf += _create_link(child, filepath, stlpath, 0.01, prefix)
        urdf += _create_joint(child, prefix)

    urdf += '</robot> \n'

    file = open(filepath + '/' + prefix + 'tetherbot.urdf','w')
    file.write(urdf)

    #print(urdf)

""" if isinstance(obj, TbRevoluteLink):
        urdf += 'revolute"> \n' \
                '  <axis xyz="0 0 1"/> \n' \
                '  <limit lower="' + str(obj.qlim[0]) + '" upper="' + str(obj.qlim[1]) + '" effort="9999" velocity="9999"/> \n'
    elif isinstance(obj, TbPrismaticLink):
        urdf += 'prismatic"> \n' \
                '  <axis xyz="0 0 1"/> \n' \
                '  <limit lower="' + str(obj.qlim[0]) + '" upper="' + str(obj.qlim[1]) + '" effort="9999" velocity="9999"/> \n'
    elif isinstance(obj, (TbGripper, TbPlatform)):
        urdf += 'floating"> \n' 
    else:
        urdf += 'fixed"> \n'  """