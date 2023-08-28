from __future__     import annotations
from ..matrices     import TransformMatrix
from ..tetherbot    import TbTetherbot
from ..tools        import isave, iload
from .Profile       import Profile
from typing         import Type, List
from abc            import ABC, abstractmethod
from datetime       import datetime
from time           import sleep
import os

class Command(ABC):

    def do(self, tetherbot: TbTetherbot, **_) -> bool:

        return False

    @abstractmethod
    def print(self) -> None:

        pass


class CommandMovePlatform(Command):

    def __init__(self, profile: Profile) -> None:

        self._targetposes = profile.poses
        self._targetpose = TransformMatrix()
        
        self._length = len(self._targetposes)
        self._current = 0

    def do(self, tetherbot: TbTetherbot, step: int = 1, **_) -> bool:

        self._targetpose = self._targetposes[self._current]
        tetherbot.platform.T_world = self._targetpose

        if self._current == self._length-1:
            return True
        else:
            self._current = min([self._current + step, self._length-1])
            return False

    def print(self) -> None:

        print('Move platform to: ' + 'x = ' + str(round(self._targetposes[-1].x,3)) + ', ' +
                                     'y = ' + str(round(self._targetposes[-1].y,3)) + ', ' +
                                     'z = ' + str(round(self._targetposes[-1].z,3)) + ', ' +
                                     'theta_x = ' + str(round(self._targetposes[-1].theta_x,2)) + ', ' +
                                     'theta_y = ' + str(round(self._targetposes[-1].tetha_y,2)) + ', ' +
                                     'theta_z = ' + str(round(self._targetposes[-1].tetha_z,2)))


class CommandMoveArm(Command):

    def __init__(self, profile: Profile) -> None:

        self._targetposes = profile.poses
        self._targetpose = TransformMatrix()

        self._length = len(self._targetposes)
        self._current = 0

    def do(self, tetherbot: TbTetherbot, step: int = 1, **_) -> bool:

        self._targetpose = self._targetposes[self._current]
        tetherbot.platform.arm.qs = tetherbot.platform.arm.ivk(self._targetpose)

        if self._current == self._length-1:
            return True
        else:
            self._current = min([self._current + step, self._length-1])
            return False

    def print(self) -> None:

        print('Move arm to: ' + 'x = ' + str(round(self._targetposes[-1].x,3)) + ', ' +
                                'y = ' + str(round(self._targetposes[-1].y,3)) + ', ' +
                                'z = ' + str(round(self._targetposes[-1].z,3)))


class CommandPickGripper(Command):

    def __init__(self, grip_idx: int =  None) -> None:

        self._grip_idx = grip_idx

    def do(self, tetherbot: TbTetherbot, **_) -> bool:

        tetherbot.pick(self._grip_idx)
        tetherbot.tension(self._grip_idx, False)

        return True

    def print(self) -> None:

        print('Pick gripper: gripper_idx = ' + str(self._grip_idx))


class CommandPlaceGripper(Command):

    def __init__(self, grip_idx: int = None, hold_idx: int = None) -> None:

        self._grip_idx = grip_idx
        self._hold_idx = hold_idx

    def do(self, tetherbot: TbTetherbot, **_) -> bool:

        tetherbot.place(grip_idx = self._grip_idx, hold_idx = self._hold_idx, correct_pose = False)
        tetherbot.tension(self._grip_idx, True)

        return True

    def print(self) -> None:

        print('Place gripper: gripper_idx = ' + str(self._grip_idx) + ' to: hold_idx = ' + str(self._hold_idx))


class CommandIdle(Command):

    def __init__(self, dt: float = 1) -> None:

        self._dt = dt

    def do(self, tetherbot: TbTetherbot, **_) -> bool:

        sleep(self._dt)

        return True

    def print(self) -> None:

        print('Idle for: dt = ' + str(self._dt))


class CommandList(List[Type[Command]]):

    def save(self, path: str = '', overwrite: bool = False) -> None:

        isave(self, path, 
              default_dir = os.path.dirname(os.path.abspath(__file__)) + '\data',
              default_name = datetime.now().strftime('%Y_%m_%d') + '_Commands',
              default_ext = 'p',
              overwrite = overwrite)

    @staticmethod
    def load(path: str) -> CommandList:

        commandlist: CommandList = iload(path,
                                         default_dir = os.path.dirname(os.path.abspath(__file__)) + '\data',
                                         default_ext = 'p')

        return commandlist

    def print(self) -> None:

        for item in self:
            item.print()