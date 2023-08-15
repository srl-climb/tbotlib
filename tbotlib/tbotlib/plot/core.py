from mpl_toolkits.mplot3d.art3d import Path3DCollection, Line3D, Poly3DCollection
from matplotlib.lines           import Line2D
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings(action = 'ignore', message = 'Mean of empty slice')
warnings.filterwarnings(action = 'ignore', message = 'invalid value encountered in double_scalars')

figs  = []
plots = {}
bgs   = {}
event = {'resize': False, 'mouse_pressed': False, 'mouse_released': False}

def show(block = True, animate = False):

    global figs
    global plots
    global bgs

    if block:
        animate = False

    # get figures, axes and plots
    figs = []
    for i in plt.get_fignums():
        figs.append(plt.figure(i))


    axes = {}
    for fig in figs:
        for ax in fig.get_axes():
            axes[ax] = fig
    
    plots = {}
    for ax in axes:
        for child in ax.get_children():
            if isinstance(child, (Line2D, Line3D, Path3DCollection, Poly3DCollection, plt.Text)): #, plt.Text
                plots[child] = ax
    
    for plot in plots:
        plot.set_animated(animate)

    # display window
    plt.show(block = block)

    # spin event loop to let the backend fully draw the figure at its final size so that
    #  a) we have the correctly sized and drawn background to grab
    #  b) we have a cached renderer so that 'ax.draw_artist' works
    plt.pause(0.1) 


    # get copy of entire figures without animated artist
    bgs = {}
    for fig in figs:
        bgs[fig] = fig.canvas.copy_from_bbox(fig.bbox)

    # draw the animated artist, this uses a cached renderer
    for plot, ax in plots.items():
        ax.draw_artist(plot)

    # show the result on screen
    for fig in figs:
        fig.canvas.blit(fig.bbox)

    # register events
    for fig in figs:
        fig.canvas.mpl_connect('button_press_event', mouse_pressed_event)
        fig.canvas.mpl_connect('button_release_event', mouse_release_event)
        fig.canvas.mpl_connect('resize_event', resize_event)
        #fig.canvas.mpl_connect('close_event', on_close)

def draw() -> None:
    
    global figs
    global plots
    global bgs

    # stop drawing when the user is panning or zooming
    if event['mouse_pressed']:
        
        while not event['mouse_released']:

            for fig in figs:
                fig.canvas.flush_events()
            
            plt.pause(0.01)
    
        event['mouse_pressed']  = False

    # update canvas when mouse was relased or resize orcurred
    if event['mouse_released'] or event['resize']:
        
        event['mouse_released'] = False
        event['resize']         = False
        
        for fig in figs:
            fig.canvas.draw()
            bgs[fig]  = fig.canvas.copy_from_bbox(fig.bbox)


    # reset the background back in the canvas state, screen unchanged
    for fig, bg in bgs.items():
        fig.canvas.restore_region(bg)

    # re-render the artist, updating the canvas state, but not the screen
    for plot, ax in plots.items():
        
        if isinstance(plot, (Path3DCollection, Poly3DCollection)):
            plot.do_3d_projection()
        
        ax.draw_artist(plot)

    # copy the image to the GUI state, but screen might not be changed yet
    # flush any pending GUI events, re-painting the screen if needed
    for fig in figs:
        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()


def mouse_pressed_event(*_):
    event['mouse_pressed'] = True

def mouse_release_event(*_):
    event['mouse_released'] = True

def resize_event(*_):
    event['resize'] = True

if __name__ == '__main__':

    x = np.linspace(0, 2 * np.pi, 100)

    fig, ax = plt.subplots()

    # animated=True tells matplotlib to only draw the artist when we
    # explicitly request it
    (ln,) = ax.plot(x, np.sin(x))

    VecStart_x = [0,1,3,5]
    VecStart_y = [2,2,5,5]
    VecStart_z = [0,1,1,5]
    VecEnd_x = [1,2,-1,6]
    VecEnd_y = [3,1,-2,7]
    VecEnd_z  =[1,0,4,9]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(4):
        ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i],VecEnd_y[i]],zs=[VecStart_z[i],VecEnd_z[i]])

    show(block=False, animate=True)

    for j in range(10000):
        ln.set_ydata(np.sin(x + (j / 100) * np.pi))
        draw()

    



    

