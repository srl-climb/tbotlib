
from __future__     import annotations
from typing import Iterable, Union
import numpy as np
import json

class ArangeDict():

    @staticmethod
    def create(start: Union[Iterable, np.ndarray], stop: Union[Iterable, np.ndarray], step: Union[Iterable, np.ndarray]) -> ArangeDict:
        
        start = np.array(start)
        stop = np.array(stop)
        step = np.array(step)

        assert start.shape == stop.shape and start.shape == stop.shape, "Inputs not of equal shape"

        shape = start.shape
        start = start.ravel()
        stop = stop.ravel()
        step = step.ravel()        
    
        return ArangeDict(start, stop, step, shape)
    
    @staticmethod
    def load(file: str) -> ArangeDict:

        with open(file, 'r') as stream:
            json_data = json.load(stream)
        
        start = np.array(json_data['start'])
        stop = np.array(json_data['stop'])
        step = np.array(json_data['step'])
        shape = tuple(json_data['shape'])
        data = np.array(json_data['data'])

        return ArangeDict(start, stop, step, shape, data)

    def __init__(self, start: np.ndarray, stop: np.ndarray, step: np.ndarray, shape: tuple, data: np.ndarray = None):

        self._shape = shape
        self._start = start
        self._stop = stop
        self._step = step
        self._num = ((self._stop - self._start) // self._step + 1).astype(int)      # number of steps for each dimension
        self._base = np.cumprod(self._num) // self._num                             # base for each dimension
        self._data = data                   
        self._len = np.prod(self._num)

        if self._data is None:
            self._data = np.full(self._len, np.nan).astype(bool)     # preallocate data, first columns are for keys, last column is for
        else:
            assert self._len == self._data.size , "Size mismatch"

    def _to_key(self, i: int) -> np.ndarray:

        return self._start + (i // self._base % self._num) * self._step
    
    def to_key(self, i: int) -> np.ndarray:

        return self._to_key(i).reshape(self._shape)
    
    def _to_index(self, key: np.ndarray) -> int:
        
        if np.all(key >= self._start) and np.all(key <= self._stop):
             return np.sum( (key - self._start) // self._step * self._base ).astype(int)
        else:
            return -1
        
    def to_index(self, key: np.ndarray) -> int:

        key = np.array(key)

        return self._to_index(key.ravel())

    def assign(self, key: np.ndarray, value: float) -> bool:

        i = self.to_index(key)

        if i != -1:
            self._data[i] = value
            return True
        else:
            return False

    def retrieve(self, key: np.ndarray) -> float:

        i = self.to_index(key)

        if i != -1:
            return self._data[i]
        else:
            return np.nan
        
    def len(self) -> float:

        return self._len

    def save(self, file: str) -> None:

        json_data = {'start': self._start.tolist(), 
                     'stop': self._stop.tolist(), 
                     'step': self._step.tolist(), 
                     'num': self._num.tolist(), 
                     'base': self._base.tolist(), 
                     'shape': self._shape, 
                     'data': self._data.tolist()}
        
        with open(file, 'w') as stream:
            json.dump(json_data, stream)

    def print_size(self):
        print("Memory size of data array in bytes: ", self._data.size * self._data.itemsize)
        
if __name__ == "__main__":

    start = [[0,0],[0,0]]
    stop = [[0.5,1],[1,1]]
    step = [[0.5,1],[1,1]]

    d = ArangeDict.create(start, stop, step)

    print()
    print('Test case: key to value to key')
    for i in range(d.len()):
        key = d.to_key(i)
        print(key)
        index = d.to_index(key)
        print(index)

    print()
    print('Test case: inaccurate key')
    key = [0.4,1,1,1]
    index = d._to_index(key)
    print(key)
    print(index)
    print(d._to_key(index))

    print()
    print('Test case: assign, retrieve value')
    key = [[0.4,1],[1,1]]
    d.assign(key, 0.2345)
    value = d.retrieve(key)
    print(value)

    """ print()
    print('Test case: saving and loading')
    import os
    file = os.path.join(os.path.dirname(__file__), 'test.json')
    d.save(file)
    d2 = ArangeDict.load(file)
    print(d._data)
    print(d2._data)

    print(d2.retrieve([[0.8,1],[1,1]])) """