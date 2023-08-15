from __future__ import annotations
import numpy as np

def insideout(arr: np.ndarray) -> np.ndarray:
    '''
    Sorts the elements of the array by their proximity to the center starting with the closest element.
    arr: Input 1D numpy array.
    res: Sorted array.
    '''

    num = len(arr)
    mid = int(np.ceil(num/2))

    res = np.empty(num)
    res[0::2] = np.flip(arr[:mid])
    res[1::2] = arr[mid:]

    return res