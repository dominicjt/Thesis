#import relevent libraries 
import numpy as np
from defineCrystal import L3Crystal
import legume 
import autograd.numpy as npa
from scipy.optimize import minimize
import functools
from legume.backend import backend as bd
from legume.utils import get_value

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

#i have been using k points tupples for the constraints. here is the fuction that 
#converts from the tupple to the k points 
def tuple_to_np_array(tup):

    #initalize empty columbs and add the kpoint we are optomizing
    col1=[]
    col2=[]
    col1.append(tup[0][0])
    col2.append(tup[0][1])

    #now add the next two tuples, k points forced above and below
    for sub_tup in tup[1:]:
        for elem in sub_tup:
            col1.append(elem[0])
            col2.append(elem[1])
    
    #turn into numpy array and return
    out = np.array([col1,col2])
    return(out)


#function for running GME
def rungme(phc,kpoints,gmax=0,options={},**kwargs):

    #run gm
    gme = legume.GuidedModeExp(phc, gmax=gmax)
    gme.run(kpoints=kpoints, **options)

    #return it
    return(gme)

#compute Q by averaging over k-grid
def compQ(gme,kpoints,optMode=0,**kwargs):
    avg = 0
    for ik in range(kpoints[0,:].size):
        (freq_im, _, _) = gme.compute_rad(0, [optMode])
        avg += gme.freqs[ik,optMode]/(2*freq_im[0])
    Q = avg/kpoints[0,:].size
    return(Q)

#compute V averaged over the k-grid 
#legume automatically normalizes fields so they integrate to uniy so intigral is not computed
def compV(gme,kpoints,optMode=0,**kwargs):
    V = 0
    for ik in range(kpoints[0,:].size):
        field = gme.get_field_xy('e', kind=ik, mind=optMode, z=0, component='xyz', Nx=100, Ny=100)[0]
        eps = gme.get_eps_xy(0)[0]
        fieldAbs = npa.abs(field['x'])**2+npa.abs(field['y'])**2+npa.abs(field['z'])**2
        maxfield = npa.real(npa.max(fieldAbs*eps))
        V += 1/maxfield
    V /= kpoints[0,:].size
    return(V)

#begining of nessicary functions for alpha calcualtion 
def holeBorders(phc,phidiv=20):

    #set up some variables
    shapes = phc.layers[0].shapes
    phis = npa.linspace(0,2*npa.pi,phidiv,endpoint=False)
    cphis = npa.cos(phis);sphis = npa.sin(phis)

    #get the list of hole atributes
    holeCords = npa.zeros((len(shapes),3))
    for i,s in enumerate(shapes):
        holeCords[i] = npa.array([s.x_cent,s.y_cent,s.r])
    
    #get the coordiates of the hole borders
    borders = npa.zeros((len(shapes),phidiv,2))
    for i,h in enumerate(holeCords):
        borders[i] = npa.array([h[0]+h[2]*cphis,h[1]+h[2]*sphis]).T

        #adjust the values so that they loop in the BZ
        borders[i,:,0][borders[i,:,0]>phc.lattice.a1[0]/2] -= phc.lattice.a1[0]
        borders[i,:,1][borders[i,:,1]>phc.lattice.a2[1]/2] -= phc.lattice.a2[1]
    
    return(borders,phis,holeCords[:,2])

#get the field for arbitrary xy points 
def get_xyfield(gme,k,n,xys,field='E',z=0,component='xyz'):

    #setup
    ft = {}
    ft['x'],ft['y'],ft['z'] = gme.ft_field_xy(field,k,n,z)
    fis = {}
    _,ind_unique = npa.unique(gme.gvec,return_index=True,axis=1)

    #loop through adding the field
    for comp in component:
        if comp in ft.keys():
            if not (comp in fis.keys()):
                fis[comp] = npa.zeros(xys[:,:,0].shape,dtype=npa.complex128)
                for indg in ind_unique:
                    fis[comp] += np.sqrt(np.pi)*ft[comp][indg]*npa.exp(1j*gme.gvec[0,indg]*xys[:,:,0]+1j*gme.gvec[1,indg]*xys[:,:,1])
        else:
            raise ValueError("component can be any combination of xyz")
    
    return(fis)

#compute the polorization dot field terms 
def compute_pdote(gme,phc,k,n,z,hbs,phis):
    
    #get the field components
    E = get_xyfield(gme,k,n,hbs,z=z)
    D = get_xyfield(gme,k,n,hbs,field='D',z=z,component='xy')

    #get the parrallel E components and the perpendicular D components
    #currently this ignores the z components (need to ask toni)
    Epara = npa.array([-npa.sin(phis)*E['x']/np.sqrt(2),npa.cos(phis)*E['y']/np.sqrt(2),E['z']/np.sqrt(2)])
    Dperp = npa.array([npa.cos(phis)*D['x'],npa.sin(phis)*D['y'],npa.zeros_like(E['z'])])

    #get the neccicary coefficents 
    eps_b = phc.layers[0].eps_b
    eps_h = 1

    #polorizability 
    p = Epara+(eps_b+eps_h)*Dperp/(2*eps_b*eps_h)

    #now take th polorizability dotted with E
    pde_r = npa.conj(E['x'])*npa.conj(p[0])+npa.conj(E['y'])*npa.conj(p[1])+npa.conj(E['z'])*npa.conj(p[2])
    pde_rp = (E['x']*p[0]+E['y']*p[1]+E['z']*p[2])
    
    return(pde_r,pde_rp)


#comput alpha/ng^2
def compute_alphaDng(gme,phc,k,n,a,z,phidiv=15,lp=40,sig=3):

    #get the hole bourders
    hbs, phis, hole_rs = holeBorders(phc,phidiv=phidiv)

    #get the pdotE terms 
    pde_r,pde_rp = compute_pdote(gme,phc,k,n,z,hbs,phis)

    #Do the multiplicatoin for the p dot e part. We will add jacobian determinate after
    pde_meshs = npa.array([npa.meshgrid(pde_r[i],pde_rp[i]) for i in range(pde_r.shape[0])])
    preSum_pde = pde_meshs[:,0]*pde_meshs[:,1]

    #the real exponential term
    phi_mesh,phi_p_mesh = npa.meshgrid(phis,phis)
    real_exp = (npa.abs(phi_mesh-phi_p_mesh)*(-hole_rs[:,npa.newaxis,npa.newaxis]))/(lp) #nm terms cancle on holes and lp

    #the imaginary exponential term
    x_meshs = npa.array([npa.meshgrid(hbs[i,:,0],hbs[i,:,0]) for i in range(hbs.shape[0])])
    imag_exp = 2*(np.linalg.norm(gme.kpoints[:,k]))*(x_meshs[:,0]-x_meshs[:,1]) #a terms cancle

    #we want to run intigral, inculding the jacobian determinant
    intigrand = preSum_pde*npa.exp(real_exp+1j*imag_exp)
    intigral = npa.sum(intigrand,axis=(1,2))*(hole_rs*npa.pi*2/phidiv)**2
    
    #calculate leading coeficents for each circle in sum
    #coeficents become 
    circle_coeffs = ((.3*2*npa.pi)*a*gme.freqs[k,n]*sig*(phc.layers[0].eps_b-1)/2)**2

    #compute sum and return
    out = intigral.shape[0]*circle_coeffs*npa.sum(intigral)*(z*a*2*10**-9)**2

    return(np.real(out))

#computes alpha, the powerloss from backscattering
def compalpha(gme,phc,kpoints,a=0,optMode=0,**kwards):
    alpha = compute_alphaDng(gme,phc,0,optMode,a,z=phc.layers[0].d/2)
    return(alpha)

#computes the ng values
def compvs(gme,phc,optMode=0,**kwargs):

    vs = np.zeros_like(gme.freqs[:,optMode])
    for i in range(len(gme.freqs[:,optMode])):
        Efield,_,_ = gme.get_field_xy('E',i,optMode,phc.layers[0].d/2,Nx=5,Ny=30)
        Hfield,_,_ = gme.get_field_xy('H',i,optMode,phc.layers[0].d/2,Nx=5,Ny=30)

        Efield = np.array([[Efield['x']],[Efield['y']],[Efield['z']]])
        Hfield = np.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])
    
        vs[i] = -np.sum(np.real(np.cross(np.conj(Efield),Hfield,axis=0)))*phc.lattice.a2[1]/5/30*phc.layers[0].d

    return(vs)

#Q running of GME. Have callback built in
@functools.lru_cache(maxsize=20) #cashes the inputs and outputs (causes a problem, look to legume docs for correct way)
def of_Q(params,Nx=0,Ny=0,optMode=0,kpoints=None,crystal=None,**kwargs):

    #inputs are made into tuples and expanded inorder to cache them, 
    #we now decode them. incoding found at costWrapper in this file
    dx,dy,dr,options = decoder(**kwargs)

    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)

    #do the cavity simulation
    phc,_ = crystal(Nx=Nx,Ny=Ny,dx=dx,dy=dy,dr=dr,**kwargs)
    
    #get the k points 
    if not kpoints:
        kpoints = kgrid(Nx=Nx,Ny=Ny,**kwargs)
    else:
        kpoints = np.array([[kpoints[0]],[kpoints[1]]])
    #run GME
    gme = rungme(phc,kpoints,options=options,**kwargs)

    #get the frequency of the first k-point. assume they are all equal 
    #as they should be with sufficent lattice size
    freq = gme.freqs[0,optMode]

    #compute the Q factor 
    Q = compQ(gme,kpoints,optMode=optMode,**kwargs)
    
    # We put a negative sign because we use in-built methods to *minimize* the objective function
    return -Q,freq

#V running of GME. Have callback built in
@functools.lru_cache(maxsize=20) #cashes the inputs and outputs
def of_V(params,Nx=0,Ny=0,optMode=0,crystal=None,**kwargs):

    #inputs are made into tuples and expanded inorder to cache them, 
    #we now decode them. incoding found at costWrapper in this file
    dx,dy,dr,options = decoder(**kwargs)
    
    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)
    
    #do the cavity simulation
    phc,_ = crystal(Nx=Nx,Ny=Ny,dx=dx,dy=dy,dr=dr,**kwargs)
    
    #get the k points 
    kpoints = kgrid(Nx=Nx,Ny=Ny,**kwargs)
    
    #run GME
    gme = rungme(phc,kpoints,options=options,**kwargs)

    #get the frequency of the first k-point. assume they are all equal 
    #as they should be with sufficent lattice size
    freq = gme.freqs[0,optMode]

    #compute the mode volsume  
    V = compV(gme,kpoints,optMode=optMode,**kwargs)

    #return V since it is being minized
    return V,freq


#Q/V running of GME. Have callback built in
@functools.lru_cache(maxsize=20) #cashes the inputs and outputs
def of_QV(params,Nx=0,Ny=0,optMode=0,crystal=None,**kwargs):
    #inputs are made into tuples and expanded inorder to cache them, 
    #we now decode them. incoding found at costWrapper in this file
    dx,dy,dr,options = decoder(**kwargs)
    
    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)

    #do the cavity simulation
    phc,_ = crystal(Nx=Nx,Ny=Ny,dx=dx,dy=dy,dr=dr,**kwargs)
    #get the k points 
    kpoints = kgrid(Nx=Nx,Ny=Ny,**kwargs)
    
    #run GME
    gme = rungme(phc,kpoints,options=options,**kwargs)

    #get the frequency of the first k-point. assume they are all equal 
    #as they should be with sufficent lattice size
    freq = gme.freqs[0,optMode]

    #compute the Q factor 
    Q = compQ(gme,kpoints,optMode=optMode,**kwargs)

    #compute the mode volume
    V = compV(gme,kpoints,optMode=optMode,**kwargs)


    #trying to maximize it so we put a minus sign in front
    return -Q/V,freq

#alpha running of GME. Have callback built in
@functools.lru_cache(maxsize=300) #cashes the inputs and outputs
def of_alphaDng(params,Nx=0,Ny=0,optMode=0,crystal=None,kpoints=None,**kwargs):
    #inputs are made into tuples and expanded inorder to cache them, 
    #we now decode them. incoding found at costWrapper in this file
    dx,dy,dr,options = decoder(**kwargs)
    
    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)

    #do the cavity simulation
    phc,_ = crystal(Nx=Nx,Ny=Ny,dx=dx,dy=dy,dr=dr,**kwargs)

    #get the k points 
    if not kpoints:
        path = kgrid(Nx=Nx,Ny=Ny,**kwargs)
    else:
        path = tuple_to_np_array(kpoints)
    
    #run GME
    options['numeig'] +=1
    gme = rungme(phc,path,options=options,**kwargs)

    #get the frequency of the first k-point. assume they are all equal 
    #as they should be with sufficent lattice size
    before = np.arange(len(kpoints[1]))+1
    after = np.arange(len(kpoints[2]))+len(before)+1
    freqs = [gme.freqs[0,optMode],gme.freqs[before,optMode].tolist(),gme.freqs[after,optMode].tolist(),
             gme.freqs[0,optMode+1],gme.freqs[before,optMode+1].tolist(),gme.freqs[after,optMode+1].tolist(),
             gme.freqs[0,optMode-1],gme.freqs[before,optMode-1].tolist(),gme.freqs[after,optMode-1].tolist()]

    #compute the alpha factor 
    alphaDng = compalpha(gme,phc,kpoints,optMode=optMode,**kwargs)

    #compute ng
    vs = compvs(gme,phc,optMode=optMode,**kwargs)
    vs = [vs[0],vs[before].tolist(),vs[after].tolist()]

    #trying to maximize it so we put a minus sign in front
    return npa.log10(alphaDng),[freqs,vs],alphaDng


#----- percell enhancement calculatoin and cost functoin 
#calculates the purcell enhancement for the field 
def calc_purcellEnhance(gme,phc,n,k,a,Nx=20,Ny=100):

    #define sigma
    sig = (1/npa.sqrt(2))*np.array([1,complex(0,1)])
    #get the group index*2pi/a
    ng = (1/(2*npa.pi))*npa.abs(npa.linalg.norm(gme.kpoints[:,k]-gme.kpoints[:,k+1],axis=0)/(gme.freqs[k,n]-gme.freqs[k+1,n]))

    #get the permiability, only looking at restricted ys
    ylim = 3*npa.sqrt(3)/2
    ys = npa.linspace(-ylim,ylim,Ny)

    #calculate leading coefficents, we cheet since it is going to be multiplied by a mask
    #we odnt care about it except outside the holes, so to save time we dont get the epsilon
    C = npa.real((ng)/((gme.freqs[k,n]**2)*npa.sqrt(phc.layers[0].eps_b)))

    #grab the field
    fields,_,_ = gme.get_field_xy('E',k,n,phc.layers[0].d/2,ygrid=ys,Nx=Nx,component='xy')

    #preform the dot product
    xcomp = sig[0]*fields['x']
    ycompPos = npa.conj(sig[1])*fields['y']
    ycompNeg = sig[1]*fields['y']
    fieldcompPos = npa.abs(xcomp+ycompPos)**2
    fieldcompNeg = npa.abs(xcomp+ycompNeg)**2

    #build mask
    xgrid,ygrid = npa.meshgrid(npa.linspace(-.5,.5,Nx),ys)
    mask = npa.ones_like(xgrid)
    for s in phc.layers[0].shapes:

        #we only care about the center few since the qd should be in the waveguide
        if npa.abs(s.y_cent)>=ylim+1:
            continue
        dist = npa.sqrt((s.x_cent-xgrid)**2+(s.y_cent-ygrid)**2)-s.r

        #take care of the case where they loop around the unit cell
        if s.x_cent<=0:
            dist1 = npa.sqrt((s.x_cent-1-xgrid)**2+(s.y_cent-ygrid)**2)-s.r
        elif s.x_cent>0:
            dist1 = npa.sqrt((s.x_cent-1-xgrid)**2+(s.y_cent-ygrid)**2)-s.r
        dist = npa.minimum(dist,dist1)

        #calculate the mask
        mask *= 1/(1+npa.exp(-74*(dist-30/a)))
    
    #get pos and neg Fs
    Fpos = C*fieldcompPos
    Fneg = C*fieldcompNeg
    return(Fpos,Fneg,mask)

#calculate the first figure of merrit from erics paper
def compMaskedCiral(gme,phc,optMode=0,a=403,**kwargs):
    fp,fn,mask = calc_purcellEnhance(gme,phc,optMode,0,a)
    forwardMasked = mask*(fp*((fp-fn)/(fn+fp)))
    return(-npa.max(forwardMasked))

#the cost funciton for the directional chiral effect
@functools.lru_cache(maxsize=20) #cashes the inputs and outputs
def of_maskedChiral(params,Nx=0,Ny=0,optMode=0,crystal=None,kpoints=None,**kwargs):
    #inputs are made into tuples and expanded inorder to cache them, 
    #we now decode them. incoding found at costWrapper in this file
    dx,dy,dr,options = decoder(**kwargs)
    
    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)

    #do the cavity simulation
    phc,_ = crystal(Nx=Nx,Ny=Ny,dx=dx,dy=dy,dr=dr,**kwargs)

    #get the k points 
    if not kpoints:
        kpoints = kgrid(Nx=Nx,Ny=Ny,**kwargs)
    else:
        kpoints = np.array([[kpoints[0],kpoints[0]+.001],[kpoints[1],kpoints[1]]])
    
    #run GME
    gme = rungme(phc,kpoints,options=options,**kwargs)

    #get the frequency of the first k-point. assume they are all equal 
    #as they should be with sufficent lattice size
    freq = gme.freqs[0,optMode]

    #compute the alpha factor 
    maskedChiral = compMaskedCiral(gme,phc,optMode=optMode,**kwargs)

    #trying to maximize it so we put a minus sign in front
    return maskedChiral,freq,maskedChiral

#computes the first 3 derivatives of the band
def compute_derivatives(gme,optMode=20,**kwargs):

    d1 = (gme.freqs[2,optMode]-gme.freqs[1,optMode])/.001
    d2 = (gme.freqs[0,optMode]+gme.freqs[2,optMode]-2*gme.freqs[1,optMode])/(.001**2)
    d3 = (-gme.freqs[0,optMode]+3*gme.freqs[1,optMode]-3*gme.freqs[2,optMode]+gme.freqs[3,optMode])/(.001**3)
    return(d1,d2,d3)


#the cost funciton for derivative controll
@functools.lru_cache(maxsize=20) #cashes the inputs and outputs
def of_d(params,Nx=0,Ny=0,optMode=0,crystal=None,kpoints=None,**kwargs):
    #inputs are made into tuples and expanded inorder to cache them, 
    #we now decode them. incoding found at costWrapper in this file
    dx,dy,dr,options = decoder(**kwargs)
    
    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)

    #do the cavity simulation
    phc,_ = crystal(Nx=Nx,Ny=Ny,dx=dx,dy=dy,dr=dr,**kwargs)

    #get the k points 
    if not kpoints:
        kpoints = kgrid(Nx=Nx,Ny=Ny,**kwargs)
    else:
        kpoints = np.array([[kpoints[0]-.001,kpoints[0],kpoints[0]+.001,kpoints[0]+.002],[kpoints[1],kpoints[1],kpoints[1],kpoints[1]]])
    
    #run GME
    gme = rungme(phc,kpoints,options=options,**kwargs)

    #get the frequency of the first k-point. assume they are all equal 
    #as they should be with sufficent lattice size
    freq = gme.freqs[0,optMode]

    #compute the derivatives
    d1,d2,d3 = compute_derivatives(gme,optMode=optMode,**kwargs)

    #trying to maximize it so we put a minus sign in front
    return d1**2+d2**2-d3**2,freq,[d1,d2,d3]

#forces band to be increasing
def bandForce(gme,optMode=0,forceParams=[0,0,0],**kwargs):
    cost = 0
    alpha = forceParams[1]
    beta = forceParams[2]
    for i in range(forceParams[0]-1):
        cost += 100*np.log(1+np.exp(-alpha*(gme.freqs[i+2,optMode]-gme.freqs[i+1,optMode]-beta)))/alpha

    return(cost)

#the cost funciton for line structure controll directly
@functools.lru_cache(maxsize=20) #cashes the inputs and outputs
def of_force(params,Nx=0,Ny=0,optMode=0,crystal=None,kpoints=None,forceParams=(0,0,0),**kwargs):
    #inputs are made into tuples and expanded inorder to cache them, 
    #we now decode them. incoding found at costWrapper in this file
    dx,dy,dr,options = decoder(**kwargs)
    
    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)

    #do the cavity simulation
    phc,_ = crystal(Nx=Nx,Ny=Ny,dx=dx,dy=dy,dr=dr,**kwargs)

    #get the k points 
    if not kpoints:
        kpoints = kgrid(Nx=Nx,Ny=Ny,**kwargs)
    else:
        path = np.vstack((np.linspace(np.pi*.5, np.pi, forceParams[0]), np.zeros(forceParams[0]))) #k vectors
        kpoints = np.array([[kpoints[0]],[kpoints[1]]])
        kpoints = np.append(kpoints,path,axis=1)
    
    #run GME
    gme = rungme(phc,kpoints,options=options,**kwargs)

    #get the frequency of the first k-point. assume they are all equal 
    #as they should be with sufficent lattice size
    freq = gme.freqs[0,optMode]
    #compute the derivatives
    cost = bandForce(gme,optMode=optMode,forceParams=forceParams)

    #trying to maximize it so we put a minus sign in front
    return cost,freq,[cost]

#This is a dummy cost function that returns zero for all values 
@functools.lru_cache(maxsize=20) #cashes the inputs and outputs
def of_monoIncreaseConstr(params,Nx=0,Ny=0,optMode=0,crystal=None,kpoints=None,**kwargs):
    #inputs are made into tuples and expanded inorder to cache them, 
    #we now decode them. incoding found at costWrapper in this file
    dx,dy,dr,options = decoder(**kwargs)
    
    #add parameters to dx, dy, and dr dictionaries 
    dx,dy,dr = placeParams(params,dx=dx,dy=dy,dr=dr)

    #do the cavity simulation
    phc,_ = crystal(Nx=Nx,Ny=Ny,dx=dx,dy=dy,dr=dr,**kwargs)

    nk = 25 # Number of k-points
    kmin, kmax = np.pi*.5, np.pi # Min and max k-values
    path = np.vstack((np.linspace(kmin, kmax, nk), np.zeros(nk)))

    #generate relivant k points
    kpoints = path[:,[8,4,12,18,25]]
    
    #run GME
    gme = rungme(phc,kpoints,options=options,**kwargs)

    freqs = [gme.freqs[0,optMode],[gme.freqs[1,optMode]],[gme.freqs[2,optMode],gme.freqs[3,optMode],gme.freqs[4,optMode]]]

    #trying to maximize it so we put a minus sign in front
    return 0,freqs,freqs



#function to wrap the cpst functons. This returns the frequency if asked for 
#and allows the cost function to be cached, speeding up run time. The chaching is used
#for the callback function and the constraint functions
def costWrapper(params, objective_function=of_Q,returnFreq=False,returnExtra=False,dx={},dy={},dr={},Nx=0,Ny=0,nk=0,gmax=2,options={},n_slab=0,dslab=0,ra=0,crystal=None,optMode=0,kpoints=None,a=0,forceParams=(0,0,0),bandwidth=0,**kwargs):
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
    if returnFreq:
        out = objective_function(paramsP,dxKeys=dxKeys,dxVals=dxVals,dyKeys=dyKeys,dyVals=dyVals,drKeys=drKeys,drVals=drVals,Nx=Nx,Ny=Ny,nk=nk,gmax=gmax,optionsKeys=optionsKeys,optionsVals=optionsVals,n_slab=n_slab,dslab=dslab,ra=ra,crystal=crystal,optMode=optMode,kpoints=kpoints,a=a,forceParams=forceParams,bandwidth=bandwidth)[:2]
        return(out)
    elif returnExtra:
        return(objective_function(paramsP,dxKeys=dxKeys,dxVals=dxVals,dyKeys=dyKeys,dyVals=dyVals,drKeys=drKeys,drVals=drVals,Nx=Nx,Ny=Ny,nk=nk,gmax=gmax,optionsKeys=optionsKeys,optionsVals=optionsVals,n_slab=n_slab,dslab=dslab,ra=ra,crystal=crystal,optMode=optMode,kpoints=kpoints,a=a,forceParams=forceParams,bandwidth=bandwidth))
    else:
        return(objective_function(paramsP,dxKeys=dxKeys,dxVals=dxVals,dyKeys=dyKeys,dyVals=dyVals,drKeys=drKeys,drVals=drVals,Nx=Nx,Ny=Ny,nk=nk,gmax=gmax,optionsKeys=optionsKeys,optionsVals=optionsVals,n_slab=n_slab,dslab=dslab,ra=ra,crystal=crystal,optMode=optMode,kpoints=kpoints,a=a,forceParams=forceParams,bandwidth=bandwidth)[0])


#inverse design for running LBFGS-B, SLSQP, and trust-const. I would like to attempt a 
#primal-log barrier method, but no good implementation seems to exist in python.
#for more complicated implementations it might be worth looking into pyomo
def ID(objective_function=of_Q,method='l-bfgs-b',dx={},dy={},dr={},constraints=None,bounds=None,callback=None,**kwargs):

    #set up the parameters 
    params = np.zeros(len(dx)+len(dy)+len(dr))
    for i,v in enumerate(dx.values()):
        params[i] = v
    for i,v in enumerate(dy.values()):
        params[i+len(dx)] = v
    for i,v in enumerate(dr.values()):
        params[i+len(dx)+len(dy)] = v


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
                        bounds=bounds, constraints=constraints, callback=callback, options={'xtol':1e-4,'gtol': 1e-9,'verbose':3,'barrier_tol':1e-9})
    else:
        result = minimize(objective_function_wrapper, params, method=method,
                        bounds=bounds, constraints=constraints, callback=callback)


    return result