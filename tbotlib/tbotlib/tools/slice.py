from __future__ import annotations
from typing     import Tuple
import numpy as np

def slice(a: np.ndarray, return_blocks: bool = True) -> Tuple[np.ndarray, np.ndarray]:

    '''
    Slices an ndarray into blocks with adjacent different columns.

    a:              nd numpy array
    return_blocks:  toggle the return of b on and off
    b:              list of blocks with adjacent different columns
    c:              first and last column index of the blocks in a
    '''

    # find adjacent different columns
    adc       = np.ones(a.shape[1]+2, dtype=bool)
    adc[1:-2] = np.all(np.diff(a).astype(bool) == 0, axis=0)
    # Note: adc is a boolean array
            # True indicates that the corresponding columns are equal 
            # False indicates that the corresponding columns are different

    # first and last column of the blocks
    c = np.where(adc[:-1] != adc[1:])[0]
    c = np.reshape(c, (-1,2))

    # handle case when all columns are the same
    if c.size == 0:

        c = np.zeros((1,2))

        if return_blocks == True:

            return c, []

        return c

    
    # handle standard case
    if return_blocks == True:

        b = []
        for i in c:
            b.append(a[:,i[0]:i[1]+1])

        return c, b

    return c


if __name__ == '__main__':

    a = np.array([[0,0,0,0,0,1,2,3,3,5,6,6],
                  [0,0,0,0,1,0,0,0,0,2,3,3],
                  [1,1,2,2,3,3,3,3,3,5,6,6]])
    
    #a = np.array([[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1]])
    c, b = slice(a)

    print(c)
    
    for item in b:
        print(item)