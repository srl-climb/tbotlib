from __future__ import annotations
import numpy as np
from typing import Iterable, Union

def ndrange(start: Union[Iterable, np.ndarray], stop: Union[Iterable, np.ndarray], step: Union[Iterable, np.ndarray]) -> np.ndarray:

    start = np.array(start)
    stop = np.array(stop)
    step = np.array(step)
    
    assert start.shape == stop.shape and start.shape == stop.shape, "Inputs not of equal shape"

    shape = start.shape
    start = start.ravel()
    stop = stop.ravel()
    step = step.ravel()

    num = (stop - start)//step+1
    base = np.cumprod(num)//num
    print(base)
    print(float(np.prod(num, dtype=np.int64)))
    print('shi')
    for i in range(np.prod(num, dtype=np.int64)):

        result = start + (i // base % num) * step
        
        yield result.reshape(shape)

    


if __name__ == '__main__':

    start = [[0,0],[0,0],[0,0],[0,0],[0,0]]
    stop = [[1,1],[1,1],[1,1],[1,1],[1,1]]
    step = [[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2],[0.2,0.2]]

    counter = 0
    for i in ndrange(start, stop, step):
        print(i)
        
        if counter == 3:
            break
        counter += 1

    """ for i in ndrange(start, stop, step):
        print(i) """
