from __future__ import annotations
from math       import ceil, sqrt
from screeninfo import get_monitors
import matplotlib.pyplot as plt

def tile(figs: list[plt.Figure], display: int = 0, size: list[float] = None, offset: list[float] = None, padding: list[float] = None) -> None:

    '''
    Tile the display with figures.
    size:    size of the area of the screen which is used for tiling
    offset:  [left, top], offset of the area
    padding: [left, top, right, bottom] additional space around the figures
    diplay:  display id 
    '''

    monitor = get_monitors()[display]
    n       = len(figs)
    cols    = ceil(sqrt(n))
    rows    = ceil(n/cols)
    tiles   = [(x,y) for x in range(rows) for y in range(cols)]

    if size is None: 
        width  = int(monitor.width/cols)
        height = int((monitor.height-34)/rows) #some extra margin, to avoid taskbar
    else:
        width  = size[0]
        height = size[1]

    if offset is None:
        offset = [0,0]

    if padding is None:
        padding = [0,0,0,0]
        
    for fig, tile in zip(figs, tiles): 

        fig.canvas.manager.window.setGeometry(tile[1]*width +offset[0]+padding[0]+monitor.x, 
                                                    tile[0]*height+offset[1]+padding[1]+monitor.y, 
                                                    width-padding[0]-padding[2], 
                                                    height-padding[1]-padding[3]) #left, top, width, height