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
plt.plot(xs,gme.freqs)
plt.show()
# %%
