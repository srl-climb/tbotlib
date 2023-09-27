from __future__ import annotations
from ..matrices import TransformMatrix
from ..tools    import isave, iload
from typing     import Union, Type, TypeVar
from datetime   import datetime
import numpy as np
import os

# Note: https://peps.python.org/pep-0673/
Self = TypeVar('Self', bound = 'TbObject') 

class TbObject:

    _id    = 0
    _subid = 0

    def __init__(self, parent: Type[TbObject] = None, T_local: TransformMatrix = None, name: str = '', fast_mode: bool = False, children: list[Type[TbObject]] = None) -> None:

        # count class instances including subclasses
        self._id = TbObject._id
        TbObject._id += 1

        # count class instances without subclasses
        self._subid = self.__class__._subid
        self.__class__._subid += 1
        
        if name == '':
            name = self.type[2:].lower() + str(self._subid)

        if children is None:
            children = []

        self._parent    = parent
        self._children  = [] # children are set by child.parent
        self._T_local   = TransformMatrix(T_local) 
        self._T_world   = TransformMatrix()
        self._name      = name
        self._fast_mode = fast_mode
        
        for child in children:
            child.parent = self

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # ensure that every new subclass starts counting subids form 0
        cls._subid = 0

    @property
    def parent(self) -> Union[TbObject, None]:
        
        return self._parent

    @parent.setter
    def parent(self, value: Type[TbObject]) -> None:

        # remove self from old parent's children
        if self._parent is not None:
            self._parent._remove_child(self)
       
        self._parent = value
        
        # add self to new parent's children
        if self._parent is not None:
            self._parent._add_child(self)

        # update transformations
        self._update_transforms()

    @property
    def children(self) -> list[Type[TbObject]]: 
        
        return self._children

    @property
    def id(self) -> int:
        
        return self._id

    @property
    def subid(self) -> int:
        
        return self._subid

    @property
    def name(self) -> str:
        
        return self._name

    @name.setter
    def name(self, value: str) -> None:

        self._name = value

    @property
    def type(self) -> str:
        return self.__class__.__name__

    @property
    def T_world(self) -> TransformMatrix:

        return self._T_world

    @T_world.setter
    def T_world(self, value: TransformMatrix) -> None:

        value = TransformMatrix(value)

        if self._parent is not None:
            self._T_local.T = self._parent.T_world.Tinv @ value.T
        else:
            self._T_local = value

        self._update_transforms()

    @property
    def T_local(self) -> TransformMatrix:

        return self._T_local

    @T_local.setter
    def T_local(self, value: TransformMatrix) -> None:

        self._T_local = value

        self._update_transforms()

    @property
    def r_world(self) -> np.ndarray:

        return self.T_world.r

    @property
    def r_local(self) -> np.ndarray:
        
        return self.T_local.r

    @r_local.setter
    def r_local(self, value: np.ndarray) -> None:
        
        self._T_local.set_r(value)

        self._update_transforms()

    @property
    def R_world(self) -> np.ndarray:
        
        return self.T_world.R

    @property
    def R_local(self) -> np.ndarray:
        
        return self.T_local.R

    @R_local.setter
    def R_local(self, value: np.ndarray) -> None:
        
        self._T_local.set_R(value)

        self._update_transforms()

    @property
    def fast_mode(self) -> bool:

        return self._fast_mode

    @fast_mode.setter
    def fast_mode(self, value: bool) -> None:

        self._fast_mode = bool(value)

    def _update_transforms(self) -> None:
        
        if self.parent is not None:
            self._T_world.T = self.parent._T_world.T @ self._T_local.T
        else:
            self._T_world.T = self._T_local.T
        
        for child in self._children:
            child._update_transforms()

    def _add_child(self, child_to_add: Type[TbObject]) -> None:
        
        self._children.append(child_to_add)

    def _remove_child(self, child_to_remove: Type[TbObject]) -> None:
     
        self._children[:] = [child for child in self._children if not child.id == child_to_remove.id]

    def get_all_children(self, filter_duplicates: bool = False) -> list[Type[TbObject]]:

        all_children = self.children

        for child in self.children:
            all_children = all_children + child.get_all_children()
        
        if filter_duplicates:
            return list(set(all_children))
        else:    
            return all_children

    def get_all_parents(self) -> list[Type[TbObject]]:

        all_parents = []

        parent = self.parent
        while(parent):
            all_parents.append(parent)
            parent = parent.parent

        return all_parents
        
    def print_info(self):

        print()
        print(self.name)
        print('With parent:')
        if self._parent is not None:
            print(self.parent.name)
        print('With children:')
        for child in self.children:
            print(child.name) 

    def save(self, path: str = '', overwrite: bool = False) -> None:

        isave(self, path, 
              default_dir = os.path.dirname(os.path.abspath(__file__)) + '\data',
              default_name = datetime.now().strftime('%Y_%m_%d') + '_Tetherbot',
              default_ext = 'p',
              overwrite = overwrite)

    @staticmethod
    def load(path: str) -> Self:

        obj: Self = iload(path,
                          default_dir = os.path.dirname(os.path.abspath(__file__)) + '\data',
                          default_ext = 'p')

        # ensure all geometries are updated
        obj._update_transforms()

        return obj


