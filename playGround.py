#%%
from defineCrystal import LNCrystal
import legume 
import numpy as np
import matplotlib.pyplot as plt
#%%
#set up loop constants
side = np.arange(10,30)
gmaxs = [1,1.25,1.5,1.75,2]
cavity = [3,5,7]

options = {'verbose': True, 'gradients': 'approx',
           'numeig': 257,       # get 5 eigenvalues
           'compute_im': False
          }

phc = LNCrystal(numRemoved=3,Nx=16,Ny=16,dslab=.6,n_slab=12,ra=.29)
legume.viz.eps_xy(phc)
gme = legume.GuidedModeExp(phc, gmax=2)
gme.run(kpoints=np.array([[0],[0]]), **options)

(freq_im, _, _) = gme.compute_rad(0, [16*16])
print(gme.freqs[0,16*16]/(2*freq_im[0]))
# %%
#set up loop constants
side = np.arange(10,30,2)
gmaxs = [1,1.25,1.5,1.75,2]
cavity = [3,5,7]

# Initialize a 3D array to store the results
# The shape is (len(side), len(gmax), len(cavity), 2) to store two computations for each combination
results = np.zeros((len(side), len(gmaxs), len(cavity), 2))

# Loop through each array
for i, s in enumerate(side):
    for j, g in enumerate(gmaxs):
        for k, c in enumerate(cavity):
            print(s,g,c)

            options = {'verbose': False, 'gradients': 'approx',
                       'numeig': s*s+1,       # get 5 eigenvalues
                       'compute_im': False
                       }

            phc = LNCrystal(numRemoved=c,Nx=s,Ny=s,dslab=.6,n_slab=12,ra=.29)
            gme = legume.GuidedModeExp(phc, gmax=g)
            gme.run(kpoints=np.array([[0],[0]]), **options)

            (freq_im, _, _) = gme.compute_rad(0, [s*s])
            
            
            # Save the computations to the results array
            results[i, j, k, 0] = gme.freqs[0,s*s]
            results[i, j, k, 1] = gme.freqs[0,s*s]/(2*freq_im[0])

# %%

computation2_slice = results[:8, :, 2, 0]
print(computation2_slice)
# Now, let's plot this as a heatmap
# Adjust these bounds as needed to control the color coding of your heatmap
vmin_bound = .25
vmax_bound = .27

plt.figure(figsize=(8, 6))
plt.imshow(computation2_slice, cmap='hot', interpolation='nearest', vmin=vmin_bound, vmax=vmax_bound)
plt.colorbar(label='$\omega$ Value')
plt.title(f'Heatmap of $\omega$ with L{cavity[2]}')
plt.xlabel('gmax')
plt.ylabel('Side length')
plt.xticks(ticks=np.arange(len(gmaxs)), labels=gmaxs)
plt.yticks(ticks=np.arange(len(side[:8])), labels=side[:8])

plt.show()
# %%
print(side)
# %%
