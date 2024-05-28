from __future__ import annotations
from typing import TYPE_CHECKING
from math import sqrt
from ..tools import ang3
import numpy as np

if TYPE_CHECKING:
    from ..tetherbot import TbTetherbot


class _Metric():

    def eval(self, u: np.ndarray, v: np.ndarray, *args) -> float:

        return 0.0


class L1Metric(_Metric):

    def __init__(self, scale: float = 1) -> None:
        
        self._scale = scale

    def eval(self, u: np.ndarray, v: np.ndarray, *args) -> float:

        return sum(abs(u - v)) * self._scale


class L2Metric(_Metric):

    def __init__(self, scale: float = 1) -> None:
        
        self._scale = scale

    def eval(self, u: np.ndarray, v: np.ndarray, *args) -> float:

        return sqrt(sum((u - v)**2)) * self._scale


class ConstantMetric(_Metric):

    def __init__(self, value: float = 0) -> None:

        self._value = value

    def eval(self, *args):

        return self._value
    

class TbAlignmentMetric(_Metric):

    def eval(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, tetherbot: TbTetherbot, *args) -> float:

        tetherbot.platform.T_world = tetherbot.platform.T_world.compose(u)
        
        return max(tetherbot.platform.arm.workspace_center.distance(v[:3]), \
                   tetherbot.platform.arm.workspace_center.distance(w[:3])) + \
                   ang3(tetherbot.platform.T_world.R[:,2], w[:3]-v[:3])
    

class StanceDisplacementMetric(_Metric):

    def eval(self, u: np.ndarray, v: np.ndarray, tetherbot: TbTetherbot, *args) -> float:

        u = np.array(u)
        v = np.array(v)

        i = u != -1

        return np.sum(np.linalg.norm(tetherbot.C_world[:, u[i]] - tetherbot.C_world[:, v[i]], axis=0)) / np.count_nonzero(i)
