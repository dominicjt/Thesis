#import relevent libraries 
import numpy as np
from defineCrystal import L3Crystal
import legume 
import autograd.numpy as npa
from scipy.optimize import minimize
import functools


#funciton that puts the parameters in the dx, dy, and dr dictionaries
def placeParams(params,save=False,dx={},dy={},dr={},**kwargs):
    
    #loop through all 3 dictionaries and keep track of itteration
    i=0
    for d in [dx,dy,dr]:
        
        #get the dictionary keys, since python 3.7 this is a deterministic order
        keys = list(d.keys())
        
        #loop through each key and add the parameter
        for key in keys:

            #for saving we want to turn them into integers
            if save:
                d[key] = params[i].item()
            else:
                d[key] = params[i]
            i+=1

    return(dx,dy,dr)

#builds the kpoint grid around the center point
def kgrid(Nx=0,Ny=0,nk=2,**kwargs):
    kx = np.linspace(0, np.pi/Nx, nk)
    ky = np.linspace(0, np.pi/Ny/np.sqrt(3)*2, nk)
    kxg, kyg = np.meshgrid(kx, ky)
    kxg = kxg.ravel()
    kyg = kyg.ravel()
    return(np.vstack((kxg, kyg)))

#function for running GME
def rungme(phc,kpoints,gmax=0,options={},**kwargs):

    #run gme
    gme = legume.GuidedModeExp(phc, gmax=gmax)
    gme.run(kpoints=kpoints, **options)

    #return it
    return(gme)

#compute Q by averaging over k-grid
def compQ(gme,kpoints,Nx=0,Ny=0,**kwargs):
    avg = 0
    for ik in range(kpoints[0,:].size):
        (freq_im, _, _) = gme.compute_rad(0, [Nx*Ny])
        avg += gme.freqs[ik,Nx*Ny]/(2*freq_im[0])
    Q = avg/kpoints[0,:].size
    return(Q)

#compute V averaged over the k-grid  #############(currently Incorrect)!!!!!!!!!!!!!!!!
def compV(gme,kpoints,Nx=0,Ny=0,n_slab=0,**kwargs):
    avg = 0
    for ik in range(kpoints[0,:].size):
        field = gme.get_field_xy('e', kind=ik, mind=Nx*Ny, z=0, component='xyz', Nx=200, Ny=200)[0]
        imax = npa.argmax(npa.abs(field['x'])+npa.abs(field['y'])+npa.abs(field['z']))
        avg += npa.square(npa.abs(field['x'][imax])+npa.abs(field['y'][imax])+npa.abs(field['z'][imax]))
    V = 1/(n_slab*avg/kpoints[0,:].size)
    return(V)

#Q running of GME. Have callback built in
#@functools.lru_cache(maxsize=20) #cashes the inputs and outputs (causes a problem, look to legume docs for correct way)
def of_Q(params,dx={},dy={},dr={},**kwargs):
    
    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)
    
    #do the cavity simulation
    phc = L3Crystal(dx=dx,dy=dy,dr=dr,**kwargs)
    
    #get the k points 
    kpoints = kgrid(**kwargs)
    
    #run GME
    gme = rungme(phc,kpoints,**kwargs)

    #compute the Q factor 
    Q = compQ(gme,kpoints,**kwargs)
    
    # We put a negative sign because we use in-built methods to *minimize* the objective function
    return -Q

#V running of GME. Have callback built in
#@functools.lru_cache(maxsize=20) #cashes the inputs and outputs
def of_V(params,dx={},dy={},dr={},**kwargs):

    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)

    #do the cavity simulation
    phc = L3Crystal(dx=dx,dy=dy,dr=dr,**kwargs)

    #get the k points 
    kpoints = kgrid(**kwargs)

    #run GME
    gme = rungme(phc,kpoints,**kwargs)

    #compute the mode volume  
    V = compV(gme,kpoints,**kwargs)

    #return V since it is being minized
    return V


#Q/V running of GME. Have callback built in
#@functools.lru_cache(maxsize=20) #cashes the inputs and outputs
def of_QV(params,dx={},dy={},dr={},**kwargs):

    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)

    #do the cavity simulation
    phc = L3Crystal(dx=dx,dy=dy,dr=dr,**kwargs)

    #get the k points 
    kpoints = kgrid(**kwargs)

    #run GME
    gme = rungme(phc,kpoints,**kwargs)

    #compute the Q factor 
    Q = compQ(gme,kpoints,**kwargs)

    #compute the mode volume
    V = compV(gme,kpoints,**kwargs)

    #trying to maximize it so we put a minus sign in front
    return -Q/V

#inverse design for running LBFGS-B, SLSQP, and trust-const. I would like to attempt a 
#primal-log barrier method, but no good implementation seems to exist in python.
#for more complicated implementations it might be worth looking into pyomo
def ID(objective_function=of_Q,method='l-bfgs-b',dx={},dy={},dr={},constraints=None,bounds=None,callback=None,**kwargs):

    #set up the parameters 
    params = np.zeros(len(dx)+len(dy)+len(dr))

    #Ensure the method is one of the accepted types
    if method not in ['l-bfgs-b', 'SLSQP', 'trust-constr']:
        raise ValueError("Invalid optimization method. Choose 'l-bfgs-b', 'SLSQP', or 'trust-constr'.")

    #Define the wrapper for the objective function to pass other params
    def objective_function_wrapper(x):
        return objective_function(x, dx=dx, dy=dy, dr=dr, **kwargs)
    
    #Perform the optimization
    result = minimize(objective_function_wrapper, params, method=method,
                      bounds=bounds, constraints=constraints, callback=callback, options={'gtol': 1,'verbose':3})

    return result