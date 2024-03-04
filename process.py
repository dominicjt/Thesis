#import relevent libraries 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


#function that shows plot of crystal with each hole labeled with its coordinate for L3

#function that shows plot of crystal with each hole labeled with its coordinate for topological

#function that, given a crystal, produces a plot of the crystal with or without the field

#function that given a path, returns an animation or single frame of the crystal with or without a field 

#produce bar graphs to compare different results

#produce supercell heat map 

def crystalPlot(phc,fill=True,plot=True,text=True):

    # Create a new figure and axis with a specified figure size
    fig, ax = plt.subplots(figsize=(phc.lattice.a1[0], phc.lattice.a2[1]))  # Here, the figure size is set to 6x6 inches

    #find the limits of the plot    
    latX = [-phc.lattice.a1[0]/2,phc.lattice.a1[0]/2]
    latY = [-phc.lattice.a2[1]/2,phc.lattice.a2[1]/2]

    #add each cirle in the plot
    for i in range(len(phc.layers[0].shapes)):
        shape = phc.layers[0].shapes[i]

        #find the center points of the cirle accounting for the wrapping 
        xcenter = (shape.x_cent+latX[1])%(2*latX[1])-latX[1]
        ycenter = (shape.y_cent+latY[1])%(2*latY[1])-latY[1]

        #add the given circle
        circle = patches.Circle((xcenter,ycenter),fill=fill, radius=shape.r, fc='black')
        ax.add_patch(circle)

        #account for the edge cases where it wraps around
        #edge case for x
        if shape.x_cent == latX[1]:
            circle = patches.Circle((-xcenter,ycenter),fill=fill, radius=shape.r, fc='black')
            ax.add_patch(circle)
    
        #edge case for y
        if shape.y_cent == latY[1]:
            circle = patches.Circle((xcenter,-ycenter),fill=fill, radius=shape.r, fc='black')
            ax.add_patch(circle)

        #edge case for x,y corner
        if shape.x_cent == latX[1] and shape.y_cent == latY[1]:
            circle = patches.Circle((-xcenter,-ycenter),fill=fill, radius=shape.r, fc='black')
            ax.add_patch(circle)

    #add the limits to the plot
    plt.xlim(*latX)
    plt.ylim(*latY)

    # Display the plot
    if plot:
        plt.show()
    else:
        return(fig,ax)

def fieldPlot(phc,gme,gapIndex=0,kIndex=0,field='E',resolution=100,title='',cbarShow=True,save=False,path=''):

    #get the plot of the phc to add the plot ontop of
    fig,ax = crystalPlot(phc,plot=False,fill=False,text=False)

    # Define the colors
    colors = ["white", "red", "black"]
    # Create a colormap
    cmap = LinearSegmentedColormap.from_list("custom_red", colors)

    #set up variables for plotting
    xyField = gme.get_field_xy(field,kIndex,gapIndex,0,Nx=resolution,Ny=resolution)
    extent = [xyField[1][0],xyField[1][-1],xyField[2][0],xyField[2][-1]]

    #set central color to zero
    #norm = TwoSlopeNorm(vcenter=0)

    #get the intensity of the field and plot that
    fieldI = np.abs(xyField[0]['x'])**2+np.abs(xyField[0]['y'])**2+np.abs(xyField[0]['z'])**2
    fieldI = fieldI[::-1]
    im = ax.imshow(fieldI,extent=extent,cmap=cmap)

    if cbarShow:

        #add the color bar
        cbar = fig.colorbar(im, ax=ax, aspect=3)

        # Label the color bar
        cbar.set_label(r'$|E|^2$',fontsize=100)

        # Set the ticks to only have two labels: 0 at the bottom and 'max' at the top
        cbar.set_ticks([np.min(fieldI), np.max(fieldI)])
        cbar.set_ticklabels(['0', 'Max'],fontsize=100)

    # Remove axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Optionally, remove the ticks as well
    ax.set_xticks([])
    ax.set_yticks([])

    plt.title(title, fontsize=100)

    if save:
        plt.savefig(path)
        plt.close('all')
        return()
    
    #show the plot 
    plt.show()

    #should return gap index and k index
    return(gapIndex,kIndex)