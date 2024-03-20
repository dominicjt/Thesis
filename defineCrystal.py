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
    return(phc,lattice)

#function that defines the L3 crystal caity structure with various parameters
def LNCrystal(numRemoved=3,Nx=0,Ny=0,dx={},dy={},dr={},dslab=0,n_slab=0,ra=0,nxbigger=0,nybigger=0,**kwargs):

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
            if iy==0 and ix>=-(numRemoved//2) and ix<=numRemoved//2:
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
    return(phc,lattice)


#function that generates the crystal structure for the topological triangle cavity
def TopoCav(Nx=0,Ny=0,sideLength=7,dx={},dy={},dr={},dslab=170/266,n_slab=11.6,ra=125/(2*266),ra1=56/(266*2),nxbigger=0,nybigger=0,**kwargs):

    r1 = ra
    r0 = ra1
    #set up lattice
    lattice = legume.Lattice([Nx, 0], [0, Ny*np.sqrt(3)/2])
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab, eps_b=n_slab)

    #for the start, lets just make the grid
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
            
            if iy%2==1:
                x = ix + .5
            else:
                x = ix
            y = iy*np.sqrt(3)/2

            #ensure that it is centered 
            if sideLength%4==3:
                x += .5

            #make new variables for r that have the flip check applied
            r0FC = r0
            r1FC = r1

            #get the side length for each iy. We can ignore values too large,
            #they are chopped off later
            width = iy+sideLength//2+1

            #value for edcases
            v = 0
            if (sideLength%4==1 and ix<0 and iy%2==1) or (sideLength%4==3 and ix>=0 and iy%2==0):
                v = -1
            elif (sideLength%4==1 and ix<0 and iy%2==0) or (sideLength%4==3 and ix>=0 and iy%2==1):
                v = 1

            #check for flipping the hole sizes
            if iy > sideLength//2 or np.abs(ix+(sideLength%4)/4)+v*.5 > (width-1)//2+.25:
                r0FC = r1
                r1FC = r0

            #start with the small holes, get the offset values and add circle
            sx = x+dx.get((ix,iy,0),0)
            sy = y+dy.get((ix,iy,0),0)
            sr = r0FC+dr.get((ix,iy,0),0)
            phc.add_shape(legume.Circle(x_cent=sx,y_cent=sy,r=sr))

            #check for making large hole small, at diagonal edges
            if np.abs(ix+(sideLength%4)/4)+v*.5 == (width-1)//2+.25:
                r1FC = r0

            #next is big holes, get the offset values and add circle
            bx = x+dx.get((ix,iy,1),0)
            by = y+dy.get((ix,iy,1),0)-np.sqrt(3)/3
            br = r1FC+dr.get((ix,iy,1),0)
            phc.add_shape(legume.Circle(x_cent=bx,y_cent=by,r=br))

    #return the crystal
    return(phc,lattice)

#function that generates the crystal structure for the topological triangle cavity
def TopoCrystal(Nx=0,Ny=0,sideLength=7,dx={},dy={},dr={},dslab=170/266,n_slab=11.6,r1=125/(2*266),r0=56/(266*2),nxbigger=0,nybigger=0,**kwargs):

    #set up lattice
    lattice = legume.Lattice([Nx, 0], [0, Ny*np.sqrt(3)/2])
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab, eps_b=n_slab)

    #for the start, lets just make the grid
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
            
            if iy%2==1:
                x = ix + .5
            else:
                x = ix
            y = iy*np.sqrt(3)/2

            #make new variables for r that have the flip check applied
            r0FC = r0
            r1FC = r1

            #start with the small holes, get the offset values and add circle
            sx = x+dx.get((ix,iy,0),0)
            sy = y+dy.get((ix,iy,0),0)
            sr = r0FC+dr.get((ix,iy,0),0)
            phc.add_shape(legume.Circle(x_cent=sx,y_cent=sy,r=sr))

            #next is big holes, get the offset values and add circle
            bx = x+dx.get((ix,iy,0),0)
            by = y+dy.get((ix,iy,0),0)-np.sqrt(3)/3
            br = r1FC+dr.get((ix,iy,0),0)
            phc.add_shape(legume.Circle(x_cent=bx,y_cent=by,r=br))

    #return the crystal
    return(phc,lattice)
    


#function that generates the crystal structure for the topological triangle cavity
def TopoWave(Nx=1,Ny=0,dx={},dy={},dr={},dslab=.571,n_slab=12.04,r1=.105,r0=.235,nxbigger=0,nybigger=0,**kwargs):

    #set up lattice
    lattice = legume.Lattice([Nx, 0], [0, Ny*np.sqrt(3)/2])
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=dslab, eps_b=n_slab)

    #for the start, lets just make the grid
    for iy in np.arange(Ny+1)-Ny//2:
        for ix in range(Nx):
            
            if iy%2==1:
                x = ix + .5
            else:
                x = ix
            y = iy*np.sqrt(3)/2-np.sqrt(3)/12

            #make new variables for r that have the flip check applied
            r0FC = r0
            r1FC = r1

            if iy>0:
                  r0FC = r1
                  r1FC = r0

            if iy != Ny//2+1:
                  #start with the small holes, get the offset values and add circle
                  sx = x+dx.get((ix,iy,0),0)
                  sy = y+dy.get((ix,iy,0),0)
                  sr = r0FC+dr.get((ix,iy,0),0)
                  phc.add_shape(legume.Circle(x_cent=sx,y_cent=sy,r=sr))


            if iy != -Ny//2+1:
                  #next is big holes, get the offset values and add circle
                  bx = x+dx.get((ix,iy,0),0)
                  by = y+dy.get((ix,iy,0),0)-np.sqrt(3)/3
                  br = r1FC+dr.get((ix,iy,0),0)
                  phc.add_shape(legume.Circle(x_cent=bx,y_cent=by,r=br))

    #return the crystal
    return(phc,lattice)