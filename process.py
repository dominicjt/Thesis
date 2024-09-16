#import relevent libraries 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from defineCrystal import TopoCav
import json
import time
import os
import legume
import copy

# Function to read existing data from JSON file
def read_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Function to write data to JSON file
def write_data(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

#function that shows plot of crystal with each hole labeled with its coordinate for L3

#function that shows plot of crystal with each hole labeled with its coordinate for topological

#function that, given a crystal, produces a plot of the crystal with or without the field

#function that given a path, returns an animation or single frame of the crystal with or without a field 

#produce bar graphs to compare different results

def crystalPlot(phc,fill=True,plot=True,text=True,color={}):

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

    #generate variables
    xyField = gme.get_field_xy(field,kIndex,gapIndex,0,Nx=resolution,Ny=resolution)
    extent = [xyField[1][0],xyField[1][-1],xyField[2][0],xyField[2][-1]]
    fieldI = np.abs(xyField[0]['x'])**2+np.abs(xyField[0]['y'])**2+np.abs(xyField[0]['z'])**2
    fieldI = fieldI[::-1]

    for i in range(1):

        #get the plot of the phc to add the plot ontop of
        fig,ax = crystalPlot(phc,plot=False,fill=False,text=False)

        # Define the colors
        colors = ["white", "red", "black"]
        # Create a colormap
        cmap = LinearSegmentedColormap.from_list("custom_red", colors)

        #set central color to zero
        #norm = TwoSlopeNorm(vcenter=0)

        #get the intensity of the field and plot that
        if i==0:
            im = ax.imshow(fieldI,extent=extent,cmap=cmap)
        else:
            fieldI = 2*np.imag(np.conjugate(xyField[0]['x'])*xyField[0]['y'])/(np.abs(xyField[0]['x'])**2+np.abs(xyField[0]['y'])**2)
            plt.imshow(fieldI,extent=extent, cmap='RdBu',vmin=-1,vmax=1)

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
            if i == 0:
                plt.savefig(path+'.png')
                plt.close('all')
            if i==1:
                plt.savefig(path+'S3.png')
                plt.close('all')
                return()
        
        #show the plot 
        plt.show()

    #should return gap index and k index
    return(gapIndex,kIndex)

def fieldPlotS3(phc,gme,gapIndex=0,kIndex=0,field='E',resolution=100,title='',cbarShow=True,save=False,path=''):

    #generate variables
    xyField = gme.get_field_xy(field,kIndex,gapIndex,0,component='xy',Nx=resolution,Ny=resolution)
    extent = [xyField[1][0],xyField[1][-1],xyField[2][0],xyField[2][-1]]
    fieldI = 2*np.imag(np.conjugate(xyField[0]['x'])*xyField[0]['y'])/(np.abs(xyField[0]['x'])**2+np.abs(xyField[0]['y'])**2)
    fieldI = fieldI[::-1]

    for i in range(1):

        #get the plot of the phc to add the plot ontop of
        fig,ax = crystalPlot(phc,plot=False,fill=False,text=False)

        #set central color to zero
        #norm = TwoSlopeNorm(vcenter=0)

        #get the intensity of the field and plot that
        if i==0:
            im = ax.imshow(fieldI,extent=extent,cmap='RdBu',vmin=-1,vmax=1)
        if cbarShow:

            #add the color bar
            cbar = fig.colorbar(im, ax=ax, aspect=3)
            cbar.ax.tick_params(labelsize=70)

        # Remove axis labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Optionally, remove the ticks as well
        ax.set_xticks([])
        ax.set_yticks([])

        plt.title(title, fontsize=100)

        if save:
            if i == 0:
                plt.savefig(path+'.png')
                plt.close('all')
            if i==1:
                plt.savefig(path+'log.png')
                plt.close('all')
                return()
        
        #show the plot 
        plt.show()

    #should return gap index and k index
    return(gapIndex,kIndex)

#this function is for running convergance testing
def convergance(size,gmax,cavity,guidedModes,file_path,minfreq,maxfreq,kpoints=np.array([[0],[0]]),plot=False,res=200):
    json_path = file_path+"/data.json"
    # Loop through each array
    for i, s in enumerate(size):
        for j, g in enumerate(gmax):
            for k, c in enumerate(cavity):
                for l, gm in enumerate(guidedModes):

                    options = {'verbose': False, 'gradients': 'approx',
                               'numeig': s*s*2,       # get 5 eigenvalues
                               'compute_im': False,
                               'gmode_inds': gm
                            }
            
                    #run simulation
                    t = time.time()
                    phc, lattice = TopoCav(sideLength=c,Nx=s,Ny=s)
                    gme = legume.GuidedModeExp(phc, gmax=g)
                    gme.run(kpoints=kpoints, **options)
                    t_tot = time.time()-t
            
                    NTindices = np.where((gme.freqs[0] > minfreq) & (gme.freqs[0] < maxfreq))[0]
                    minI = NTindices[0]
                    maxI = NTindices[-1]+1

                    (freq_im, _, _) = gme.compute_rad(0,NTindices)

                    datatosave = {'time':t_tot,'gmax':g,'cavity':c,'size':s,'gmode_ins':gm,'planeWaves':int(gme.n1g*gme.n2g),
                            'indices':NTindices.tolist(),'freqs':(266/gme.freqs).tolist(),
                            'Q':(gme.freqs[0,minI:maxI]/(2*freq_im)).tolist()}
            
                    data = read_data(json_path)
                    data.append(datatosave)
                    write_data(json_path, data)

                    #save plots
                    if plot:
                        plot_path = file_path + f'/cavity{s}{c}{int(g*100)}{len(guidedModes)}'
                        os.makedirs(plot_path, exist_ok=True)
                        for i in NTindices:
                            plot(phc,gme,gapIndex=i,resolution=res,title=f'Wavelength = {np.round(266/gme.freqs[0,i],2)}, size={s}\ncavity={c}, gmax={np.round(g,2)}',cbarShow=False,save=True,path=plot_path+f'/{i}')


#Noise testing. This functlist to store data ion evaluates some parameter as the hole x,y locatoins
#are randomaly perturbed by pulling noise from a gaussain distributoin
def NoiseTest(phc,func,noiseSTD=.05,nruns=100,path='',**kwargs):

    #initalize json file to store data
    with open(path,'w') as file:
        json.dump([],file)
    
    for i in range(nruns):
        #make a copy of the crystal to add noise to 
        phcNoisy = copy.deepcopy(phc)
        for shape in phcNoisy.layers[0].shapes:
            shape.x_cent += np.random.normal(0,noiseSTD) 
            shape.y_cent += np.random.normal(0,noiseSTD) 
        
        #add the new data to the dictionary
        with open(path,'r') as file:
            data = json.load(file)
        data.append(func(phcNoisy,**kwargs))
        with open(path,'w') as file:
            json.dump(data,file,indent=4)

def Q(phc,gmax=1.5,optMode=0):
    gme = legume.GuidedModeExp(phc,gmax=gmax)
    gme.run(kpoints=np.array([[0],[0]]),numeig=optMode+1)
    (freq_im,_,_)=gme.compute_rad(0,[optMode])
    return([gme.freqs[0,optMode]/(2*freq_im[0]),gme.freqs[0,optMode]])
        
def bands(phc,gmax=1,maxMode=2000):
    gme = legume.GuidedModeExp(phc,gmax=gmax)
    gme.run(kpoints=np.array([[0],[0]]),numeig=maxMode,compute_im=False)
    return(gme.freqs[0].tolist())