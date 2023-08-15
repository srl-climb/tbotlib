from __future__ import annotations
import numpy as np

def interleave(a: np.ndarray, b: np.ndarray, order:bool=0, stagger:bool=True, fill:bool=True) -> np.ndarray:
    '''
    interleaves the arrays of in list a and b
    a:       list of numpy arrays
    b:       list of numpy arrays
    order:   specifies if an array of a (0) or b (1) comes first
    stagger: if true, the arrays are interleaved and staggered, gaps are filled with zeros
    fill:    if true, the gaps are filled with the last column of the left adjecent array
    '''

    # number of rows of each array in a and b
    k_a = np.shape((np.atleast_2d(a[0])))[0]
    k_b = np.shape((np.atleast_2d(b[0])))[0] 

    # number of columns of each array in a and b
    n_a = []
    for arr in a:
        n_a.append(np.shape((np.atleast_2d(arr)))[1])

    n_b = []
    for arr in b:
        n_b.append(np.shape((np.atleast_2d(arr)))[1])

    # order of a and b in the interleaved array 
    ord_a = (order == 1) + np.arange(len(a)) * 2
    ord_b = (order == 0) + np.arange(len(b)) * 2

    # first column indices in the interleaved array
    itl = np.empty(len(n_a)+len(n_b), dtype=np.int16)
    itl[ord_a] = n_a
    itl[ord_b] = n_b
    itl = np.cumsum(itl) - itl

    # first column indicies of a and b in the interleaved array
    itl_a = itl[ord_a]
    itl_b = itl[ord_b]

    # create interleaved array
    if stagger == True:

        c = np.zeros((k_a+k_b, sum(n_a)+sum(n_b)))

        if fill == True:
            c[:k_a,:] = np.transpose([np.atleast_2d(a[0])[:,0]])
            c[k_a:,:] = np.transpose([np.atleast_2d(b[0])[:,0]])

        for i in range(len(n_a)):
            c[:k_a, itl_a[i]:itl_a[i]+n_a[i]] = a[i]

            if fill == True:
                c[:k_a, itl_a[i]+n_a[i]:] = np.transpose([np.atleast_2d(a[i])[:,-1]])

        for i in range(len(n_b)):
            c[k_a:, itl_b[i]:itl_b[i]+n_b[i]] = b[i]

            if fill == True:
                c[k_a:, itl_b[i]+n_b[i]:] = np.transpose([np.atleast_2d(b[i])[:,-1]])

    else:

        c = np.zeros((k_a, sum(n_a)+sum(n_b)))

        for i in range(len(n_a)):
            c[:, itl_a[i]:itl_a[i]+n_a[i]] = a[i]

        for i in range(len(n_b)):
            c[:, itl_b[i]:itl_b[i]+n_b[i]] = b[i]

    return c


if __name__ == '__main__':

    a = [np.array((1)),
         np.array((3,3,3)),
         np.array((4,4,4,4))]
    b = [np.array((5,5,5,5,5)),
         np.array((2,2)),
         np.array((3,3,3))]

    c = interleave(a, b, order=1, stagger=True, fill=True)
    print(a)
    print(b)
    print(c)

