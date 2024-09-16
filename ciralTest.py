#%%
from defineCrystal import TopoWave
import legume
import numpy as np
import matplotlib.pyplot as plt
#%%
#toni's crystal
a = 266 # nm, the lattice constant (pitch) of the crystal
h = 170 # nm, the height (depth) of the slab
r1, r2 = .235*a, .105*a # nm, starting radii

nk = 200 # Number of k-points
kmin, kmax = np.pi*.5, np.pi # Min and max k-values
path = np.vstack((np.linspace(kmin, kmax, nk), np.zeros(nk))) #k vectors

phc,_ = TopoWave(Ny=21,n_slab=3.4638**2,r0=r2/a,r1=r1/a,dslab=h/a)
legume.viz.eps_xy(phc,Nx=200,Ny=200)
#%%
gme = legume.GuidedModeExp(phc,gmax=2)
gme.run(kpoints=path,numeig=50,compute_im=False)
#%%
xs = np.linspace(kmin,kmax,nk)/(2*np.pi)
plt.plot(xs,gme.freqs[:,:50])
plt.show()
#%%
legume.viz.field(gme,'E',63,20,z=0,component='x')
plt.show()
legume.viz.field(gme,'E',63,21,z=0,component='y')
plt.show()
#%%
#calculates the purcell enhancement for the field 
def purcellEnhance(gme,n,k,Nx=20,Ny=100):

    #define sigma
    sig = (1/np.sqrt(2))*np.array([1,complex(0,1)])
    #get the group index*2pi/a
    ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gme.kpoints[:,k]-gme.kpoints[:,k-1],axis=0)/(gme.freqs[k,n]-gme.freqs[k-1,n]))

    #get the permiability, only looking at restricted ys
    ylim = 3*np.sqrt(3)/2
    ys = np.linspace(-ylim,ylim,Ny)
    eps,_,_ = gme.get_eps_xy(h/(2*a),ygrid=ys,Nx=Nx)

    #calculate leading coefficents
    C = np.real((3*np.pi**2*(3E8)**2*ng)/(gme.freqs[k,n]*np.sqrt(eps)))

    #grab the field
    fields,_,_ = gme.get_field_xy('E',k,n,h/(a*2),ygrid=ys,Nx=Nx,component='xy')

    #preform the dot product
    xcomp = sig[0]*fields['x']
    ycompPos = np.conj(sig[1])*fields['y']
    ycompNeg = sig[1]*fields['y']
    fieldcompPos = np.abs(xcomp+ycompPos)**2
    fieldcompNeg = np.abs(xcomp+ycompNeg)**2

    #build mask
    ys = int(Ny*21/6)
    mask = legume.viz.eps_xy(phc,Nx=Nx,Ny=ys,plot=False)
    mask = (mask[int(ys*(7.5/21)):int(ys*(13.5/21)),:]-1)/np.max(mask)

    #get pos and neg Fs
    Fpos = C*fieldcompPos
    Fneg = C*fieldcompNeg
    return(Fpos*mask,Fneg*mask)

def concurrence(gme,n,k,S=0,tau=0,Nx=20,Ny=100):

    #get the field
    ylim = 3*np.sqrt(3)/2
    ys = np.linspace(-ylim,ylim,Ny)
    fields,_,_ = gme.get_field_xy('E',k,n,h/(a*2),ygrid=ys,Nx=Nx,component='xy')

    #calc phi
    phi = np.arctan(np.imag(fields['x'])/np.real(fields['x']))-np.arctan(np.imag(fields['y'])/np.real(fields['y']))
    
    #calc concurrence
    C = np.sin(phi)**2/(1+np.cos(S*tau)*np.cos(phi)**2)

    return(C)


#%%
fp,fn = purcellEnhance(gme,20,63,Nx=100,Ny=200)
#%%
plt.imshow((fn-fp)/(fn+fp),'bwr')
plt.colorbar()

plt.show()
#%%
C = concurrence(gme,20,63,Nx=100,Ny=200)
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
plt.ylim(1,10**2)
plt.show()
#%%
print(np.linalg.norm(gme.kpoints[:,1:]-gme.kpoints[:,:-1],axis=0))
#%%
ys = int(200*21/6)
out = legume.viz.eps_xy(phc,Nx=100,Ny=ys,plot=False)
#%%
plt.imshow(out[int(ys*(7.5/21)):int(ys*(13.5/21)),:])
# %%
