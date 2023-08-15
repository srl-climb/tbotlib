from __future__ import annotations

def between(val: float, lim: list[float]) -> bool | float:

    '''
    Check if a value is between limits.

    Returns True and value if the value is between the limits.
    Returns False and upper limit if the value is above the upper limit.
    Returns False and lower limit if the value is below the lower limit.

    val: value to be checked
    lim: list with lower (0) and upper (1) value
    '''

    if lim[0] <= val and val <= lim[1]:

        return True, val

    elif lim[0] > val:
        return False, lim[0]

    else:
        return False, lim[1]

    