from time import time

__starttime__ = 0


def tic() -> None:

    global __starttime__ 
   
    __starttime__ = time()

    pass


def toc() -> None:

    global __starttime__

    print("--- %s seconds ---" % (time() - __starttime__))

    pass
