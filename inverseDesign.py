#import relevent libraries 
import numpy as np
from defineCrystal import L3Crystal
import legume 
import autograd.numpy as npa
from scipy.optimize import minimize
import functools

#decoder used to put dx,dy,dr, and options back into proper form. 
#they are incoded in costWrapper to allow them to be cached.
def decoder(dxKeys=(),dxVals=(),dyKeys=(),dyVals=(),drKeys=(),drVals=(),optionsKeys=(),optionsVals=(),**kwargs):

    #check how they are stored in costWrapper, but the basic idea is
    #the keys and values from each are both put in a tuple. we just put
    #them back into dictionaries and pass back 
    dx = {}; dy = {}; dr = {}; options = {}
    #dx
    for k,v in zip(dxKeys,dxVals):
        dx[k] = v
    #dy
    for k,v in zip(dyKeys,dyVals):
        dy[k] = v
    #dr
    for k,v in zip(drKeys,drVals):
        dr[k] = v
    #optoins
    for k,v in zip(optionsKeys,optionsVals):
        options[k] = v
    
    return(dx,dy,dr,options)

                   

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
#needs to make scaling of y more general
def kgrid(Nx=0,Ny=0,nk=1,**kwargs):
    kx = np.linspace(0, np.pi/Nx, nk)
    ky = np.linspace(0, np.pi/Ny/np.sqrt(3)*2, nk)
    kxg, kyg = np.meshgrid(kx, ky)
    kxg = kxg.ravel()
    kyg = kyg.ravel()
    return(np.vstack((kxg, kyg)))

#function for running GME
def rungme(phc,kpoints,gmax=0,options={},**kwargs):

    #run gm
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

#compute V averaged over the k-grid 
#legume automatically normalizes fields so they integrate to uniy so intigral is not computed
def compV(gme,kpoints,Nx=0,Ny=0,**kwargs):
    V = 0
    for ik in range(kpoints[0,:].size):
        field = gme.get_field_xy('e', kind=ik, mind=Nx*Ny, z=0, component='xyz', Nx=100, Ny=100)[0]
        eps = gme.get_eps_xy(0)
        fieldAbs = npa.abs(field['x'])**2+npa.abs(field['y'])**2+npa.abs(field['z'])**2
        maxfield = npa.max(fieldAbs*eps)
        V += 1/maxfield
    V /= kpoints**2
    return(V)

#Q running of GME. Have callback built in
@functools.lru_cache(maxsize=20) #cashes the inputs and outputs (causes a problem, look to legume docs for correct way)
def of_Q(params,Nx=0,Ny=0,**kwargs):

    #inputs are made into tuples and expanded inorder to cache them, 
    #we now decode them. incoding found at costWrapper in this file
    dx,dy,dr,options = decoder(**kwargs)
    
    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)
    
    #do the cavity simulation
    phc = L3Crystal(Nx=Nx,Ny=Ny,dx=dx,dy=dy,dr=dr,**kwargs)
    
    #get the k points 
    kpoints = kgrid(Nx=Nx,Ny=Ny,**kwargs)
    
    #run GME
    gme = rungme(phc,kpoints,options=options,**kwargs)

    #get the frequency of the first k-point. assume they are all equal 
    #as they should be with sufficent lattice size
    freq = gme.freqs[0,Nx*Ny]

    #compute the Q factor 
    Q = compQ(gme,kpoints,Nx=Nx,Ny=Ny,**kwargs)
    
    # We put a negative sign because we use in-built methods to *minimize* the objective function
    return -Q,freq

#V running of GME. Have callback built in
@functools.lru_cache(maxsize=20) #cashes the inputs and outputs
def of_V(params,Nx=0,Ny=0,**kwargs):

    #inputs are made into tuples and expanded inorder to cache them, 
    #we now decode them. incoding found at costWrapper in this file
    dx,dy,dr,options = decoder(**kwargs)
    
    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)
    
    #do the cavity simulation
    phc = L3Crystal(Nx=Nx,Ny=Ny,dx=dx,dy=dy,dr=dr,**kwargs)
    
    #get the k points 
    kpoints = kgrid(Nx=Nx,Ny=Ny,**kwargs)
    
    #run GME
    gme = rungme(phc,kpoints,options=options,**kwargs)

    #get the frequency of the first k-point. assume they are all equal 
    #as they should be with sufficent lattice size
    freq = gme.freqs[0,Nx*Ny]

    #compute the mode volsume  
    V = compV(gme,kpoints,Nx=0,Ny=0,**kwargs)

    #return V since it is being minized
    return V,freq


#Q/V running of GME. Have callback built in
@functools.lru_cache(maxsize=20) #cashes the inputs and outputs
def of_QV(params,Nx=0,Ny=0,**kwargs):

    #inputs are made into tuples and expanded inorder to cache them, 
    #we now decode them. incoding found at costWrapper in this file
    dx,dy,dr,options = decoder(**kwargs)
    
    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)
    
    #do the cavity simulation
    phc = L3Crystal(Nx=Nx,Ny=Ny,dx=dx,dy=dy,dr=dr,**kwargs)
    
    #get the k points 
    kpoints = kgrid(Nx=Nx,Ny=Ny,**kwargs)
    
    #run GME
    gme = rungme(phc,kpoints,options=options,**kwargs)

    #get the frequency of the first k-point. assume they are all equal 
    #as they should be with sufficent lattice size
    freq = gme.freqs[0,Nx*Ny]

    #compute the Q factor 
    Q = compQ(gme,kpoints,Nx=0,Ny=0,**kwargs)

    #compute the mode volume
    V = compV(gme,kpoints,Nx=0,Ny=0,**kwargs)

    #trying to maximize it so we put a minus sign in front
    return -Q/V,freq

#function to wrap the cpst functons. This returns the frequency if asked for 
#and allows the cost function to be cached, speeding up run time. The chaching is used
#for the callback function and the constraint functions
def costWrapper(params, objective_function=of_Q,returnFreq=False,dx={},dy={},dr={},Nx=0,Ny=0,nk=0,gmax=2,options={},n_slab=0,dslab=0,ra=0,**kwargs):

    #convert params to a hashable type for cahcing
    paramsP = tuple(params)

    #incode dx,dy,dr and options into tupples so they can be cached
    dxKeys = []; dxVals = []
    for key,value in dx.items():
        dxKeys.append(key)
        dxVals.append(value)
    dxKeys = tuple(dxKeys); dxVals = tuple(dxKeys)
    #dy
    dyKeys = []; dyVals = []
    for key,value in dy.items():
        dyKeys.append(key)
        dyVals.append(value)
    dyKeys = tuple(dyKeys); dyVals = tuple(dyKeys)
    #dr
    drKeys = []; drVals = []
    for key,value in dr.items():
        drKeys.append(key)
        drVals.append(value)
    drKeys = tuple(drKeys); drVals = tuple(drKeys)
    #options
    optionsKeys = []; optionsVals = []
    for key,value in options.items():
        optionsKeys.append(key)
        optionsVals.append(value)
    optionsKeys = tuple(optionsKeys); optionsVals = tuple(optionsVals)

    #call the fuction so that it can be cached
    ###### Note, if you are going to add new dependancies, you must add them here ######
    cost,freq = objective_function(paramsP,dxKeys=dxKeys,dxVals=dxVals,dyKeys=dyKeys,dyVals=dyVals,drKeys=drKeys,drVals=drVals,Nx=Nx,Ny=Ny,nk=nk,gmax=gmax,optionsKeys=optionsKeys,optionsVals=optionsVals,n_slab=n_slab,dslab=dslab,ra=ra)
    
    if returnFreq:
        return(cost,freq)
    else:
        return(cost)


#inverse design for running LBFGS-B, SLSQP, and trust-const. I would like to attempt a 
#primal-log barrier method, but no good implementation seems to exist in python.
#for more complicated implementations it might be worth looking into pyomo
def ID(objective_function=of_Q,method='l-bfgs-b',dx={},dy={},dr={},constraints=None,bounds=None,callback=None,**kwargs):

    #set up the parameters 
    params = np.zeros(len(dx)+len(dy)+len(dr))

    #Ensure the method is one of the accepted types
    if method not in ['l-bfgs-b', 'SLSQP', 'trust-constr']:
        raise ValueError("Invalid optimization method. Choose 'l-bfgs-b', 'SLSQP', or 'trust-constr'.")

    #Define the wrapper for the objective function to pass other params. there is another layer of wrapping 
    #defined above
    def objective_function_wrapper(x):
        return costWrapper(x, objective_function=objective_function, dx=dx, dy=dy, dr=dr, **kwargs)
    
    #Perform the optimization
    if method=='trust-constr':
        #handle this slightly differently, add verbose and gtol to help it converge
        result = minimize(objective_function_wrapper, params, method=method,
                        bounds=bounds, constraints=constraints, callback=callback, options={'gtol': 1,'verbose':3})
    else:
        result = minimize(objective_function_wrapper, params, method=method,
                        bounds=bounds, constraints=constraints, callback=callback)


    return result