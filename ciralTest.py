#%%
from defineCrystal import TopoWave
import legume
import numpy as np
import matplotlib.pyplot as plt
#%%
#toni's crystal
a = 403 # nm, the lattice constant (pitch) of the crystal
h = 170 # nm, the height (depth) of the slab
r1, r2 = .235*a, .105*a # nm, starting radii

nk = 25 # Number of k-points
kmin, kmax = np.pi*.5, np.pi # Min and max k-values
path = np.vstack((np.linspace(kmin, kmax, nk), np.zeros(nk))) #k vectors

phc,_ = TopoWave(Ny=21,n_slab=3.4638**2,ra=(r2/a,r1/a),dslab=h/a)
legume.viz.eps_xy(phc,Nx=200,Ny=200)
#%%
gme = legume.GuidedModeExp(phc,gmax=2)
gme.run(kpoints=path,numeig=50,compute_im=False)
#%%
xs = np.linspace(kmin,kmax,nk)/(2*np.pi)
plt.plot(xs,gme.freqs[:,:50])
plt.show()
#%%
legume.viz.field(gme,'E',12,20,z=0,component='x')
plt.show()
legume.viz.field(gme,'E',12,21,z=0,component='y')
plt.show()
#%%
#%%
#calculates the purcell enhancement for the field 
def purcellEnhance(gme,phc,n,k,Nx=20,Ny=100):

    #define sigma
    sig = (1/np.sqrt(2))*np.array([1,complex(0,1)])
    #get the group index*2pi/a
    ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gme.kpoints[:,k]-gme.kpoints[:,k-1],axis=0)/(gme.freqs[k,n]-gme.freqs[k-1,n]))

    #get the permiability, only looking at restricted ys
    ylim = 3*np.sqrt(3)/2
    ys = np.linspace(-ylim,ylim,Ny)

    #calculate leading coefficents, we cheet since it is going to be multiplied by a mask
    #we odnt care about it except outside the holes, so to save time we dont get the epsilon
    C = np.real((ng)/((gme.freqs[k,n]**2)*np.sqrt(phc.layers[0].eps_b)))

    #grab the field
    fields,_,_ = gme.get_field_xy('E',k,n,h/(a*2),ygrid=ys,Nx=Nx,component='xy')

    #preform the dot product
    xcomp = sig[0]*fields['x']
    ycompPos = np.conj(sig[1])*fields['y']
    ycompNeg = sig[1]*fields['y']
    fieldcompPos = np.abs(xcomp+ycompPos)**2
    fieldcompNeg = np.abs(xcomp+ycompNeg)**2

    #build mask
    xgrid,ygrid = np.meshgrid(np.linspace(-.5,.5,Nx),ys)
    mask = np.ones_like(xgrid)
    for s in phc.layers[0].shapes:

        #we only care about the center few since the qd should be in the waveguide
        if np.abs(s.y_cent)>=ylim+1:
            continue
        dist = np.sqrt((s.x_cent-xgrid)**2+(s.y_cent-ygrid)**2)-s.r

        #take care of the case where they loop around the unit cell
        if s.x_cent<=0:
            dist1 = np.sqrt((s.x_cent-1-xgrid)**2+(s.y_cent-ygrid)**2)-s.r
        elif s.x_cent>0:
            dist1 = np.sqrt((s.x_cent-1-xgrid)**2+(s.y_cent-ygrid)**2)-s.r
        dist = np.minimum(dist,dist1)

        #calculate the mask
        mask *= 1/(1+np.exp(-74*(dist-30/a)))
    
    #get pos and neg Fs
    Fpos = C*fieldcompPos
    Fneg = C*fieldcompNeg
    return(Fpos,Fneg,mask)

def concurrence(gme,n,k,S=0,tau=0,Nx=20,Ny=100):
    ylim = 3*np.sqrt(3)/2
    ys = np.linspace(-ylim,ylim,Ny)
    fields,_,_ = gme.get_field_xy('E',k,n,h/(a*2),ygrid=ys,Nx=Nx,component='xy')

    #calc phi
    phi = np.arctan(np.imag(fields['x'])/np.real(fields['x']))-np.arctan(np.imag(fields['y'])/np.real(fields['y']))
    
    #calc concurrence
    C = np.sin(phi)**2/(1+np.cos(S*tau)*np.cos(phi)**2)

    return(C)


#%%
fp,fn,mesh = purcellEnhance(gme,phc,20,8,Nx=100,Ny=200)

#%%
plt.imshow(mesh*fn)
plt.colorbar()
plt.show()
#%%
plt.imshow(mesh*(fp*((fp-fn)/(fn+fp))),'bwr')
plt.colorbar()
plt.show()
np.max(mesh*(fp*((fn-fp)/(fn+fp))))
#%%
C = concurrence(gme,20,12,Nx=100,Ny=200)
plt.imshow(C)
plt.show()
#%%
#the ng is working
n = 21
ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gme.kpoints[:,1:]-gme.kpoints[:,:-1],axis=0)/(gme.freqs[1:,n]-gme.freqs[:-1,n]))
plt.plot(gme.freqs[1:,n],ng)
n = 20
ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gme.kpoints[:,1:]-gme.kpoints[:,:-1],axis=0)/(gme.freqs[1:,n]-gme.freqs[:-1,n]))
plt.plot(gme.freqs[1:,n],ng)
plt.yscale('log')
plt.show()
#%%
print(ng)

# %%
