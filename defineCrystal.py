#import relevent libraries 
import legume 
import numpy as np
#dx, dy, and dr are dictionaries: {(coordiante):offset,...}

#function that defines the L3 crystal caity structure with various parameters
def L3Crystal(Nx=0,Ny=0,dx={},dy={},dr={},dslab=0,n_slab=0,ra=0,nxbigger=0,nybigger=0,**kwargs):

    #Define the lattice and crystal
    lattice = legume.Lattice([Nx, 0], [0, Ny*np.sqrt(3)/2])
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab, eps_b=n_slab)

    #add the holes
    for ittery in range(Ny+nybigger):
        for itterx in range(Nx+nxbigger):
            
            #adjust indexes the center at middle of the middle 
            if ittery>Ny//2+nybigger:
                iy = ittery-(Ny+nybigger)
            else:
                iy = ittery
            if itterx>Nx//2+nybigger:
                ix = itterx-(Nx+nxbigger)
            else:
                ix = itterx
            
            #remove 3 center holes 
            if iy==0 and (ix==-1 or ix==0 or ix==1):
                continue

            #get the x and y default values
            if iy%2==1:
                x = itterx + 0.5
            else:
                x = itterx
            y = ittery*np.sqrt(3)/2
            
            #adjust for offset values
            x += dx.get((ix,iy),0)
            y += dy.get((ix,iy),0)
            r = ra+dr.get((ix,iy),0)

            #add circle
            phc.add_shape(legume.Circle(x_cent=x,y_cent=y,r=r))

    #return the photonic crystal
    return(phc)




#function that generates the crystal structure for the topological triangle cavity

