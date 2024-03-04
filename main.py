#%%
from saveLoad import experiment
from inverseDesign import ID
from genConst import L3const
#%%


dx = {(2,0):0,(3,0):0,(4,0):0,(-2,0):0,(-3,0):0,(-4,0):0,
      (0,1):0,(1,1):0,(2,1):0,(-1,1):0,(-2,1):0,(-3,1):0,
      (0,-1):0,(1,-1):0,(2,-1):0,(-1,-1):0,(-2,-1):0,(-3,-1):0}

dy = {(2,0):0,(3,0):0,(4,0):0,(-2,0):0,(-3,0):0,(-4,0):0,
      (0,1):0,(1,1):0,(2,1):0,(-1,1):0,(-2,1):0,(-3,1):0,
      (0,-1):0,(1,-1):0,(2,-1):0,(-1,-1):0,(-2,-1):0,(-3,-1):0}

dr = {(2,0):0,(3,0):0,(4,0):0,(-2,0):0,(-3,0):0,(-4,0):0,
      (0,1):0,(1,1):0,(2,1):0,(-1,1):0,(-2,1):0,(-3,1):0,
      (0,-1):0,(1,-1):0,(2,-1):0,(-1,-1):0,(-2,-1):0,(-3,-1):0}

runs = {'name':'freqConfine',
        #'lbfgsbBig': {'dx':dx,'dy':dy,'dr':dr,'method':'l-bfgs-b'},
        'trust-constrBig': {'dx':dx,'dy':dy,'dr':dr,'method':'trust-constr','constraints':True,'minrad':.05,'mindist':.05,
                         'minfreq':.261,'maxfreq':.3,'constFunc':L3const}}

#experiment(runs,ID)


#%%
from defineCrystal import TopoWave, TopoCav, TopoCrystal
import legume
import numpy as np
import matplotlib.pyplot as plt
from process import fieldPlot
plt.rcParams.update({'font.size': 16})

#%%
phc, lattice = TopoCav(Nx = 40, Ny = 40, sideLength=11)
#phc, lattice = TopoWave(Nx = 1, Ny = 13, sideLength=21)
#phc, lattice = TopoCrystal(Nx=10,Ny=10)
legume.viz.eps_xy(phc,Nx=300,Ny=300)
#%%
# Initialize GME
gme = legume.GuidedModeExp(phc, gmax=1.5)

# Solve for the real part of the frequencies
gme.run(kpoints=np.array([[0],[0]]), verbose=True, compute_im=False,numeig = 2000)
#%%
indices = np.where((266/gme.freqs[0] > 960) & (266/gme.freqs[0] < 1080))
print(indices)
minI = indices[0][0]
maxI = indices[0][-1]+1

#compute Q
(freq_im, _, _) = gme.compute_rad(0,indices[0])
Qs = gme.freqs[0,minI:maxI]/(2*freq_im)
#%%
plt.plot(266/gme.freqs[0,minI:maxI],Qs,'o')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Q')
plt.show()
# %%
indicesClose = np.where((266/gme.freqs[0] > 1005) & (266/gme.freqs[0] < 1035))
minIclose = indicesClose[0][0]
maxIclose = indicesClose[0][-1]+1
print(minI)

plt.plot(266/gme.freqs[0,minIclose:maxIclose],Qs[:maxIclose-minIclose],'o')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Q')
plt.show()

#%%
fieldPlot(phc,gme,gapIndex=1317,resolution=300,title=f'Wavelength = {np.round(266/gme.freqs[0,1317],2)}, Q = {np.round(Qs[1317-minI],2)}',cbarShow=False)
# %%
print(np.where(Qs == np.max(Qs)))
# %%
print(np.abs(266/gme.freqs[0] - 1018).argmin())

# %%
indices = np.where((266/gme.freqs[0] > 980) & (266/gme.freqs[0] < 1030))
print(indices)
minI = indices[0][0]
maxI = indices[0][-1]+1
#%%
for i in np.arange(minI, maxI):
    print(i)
    fieldPlot(phc,gme,gapIndex=i,resolution=300,title=f'Wavelength = {np.round(266/gme.freqs[0,i],2)}, Q = {np.round(Qs[i-minI],2)}',cbarShow=False)
# %%
fieldPlot(phc,gme,gapIndex=1612,resolution=500,title=f'Wavelength = {np.round(266/gme.freqs[0,1612],2)}, Q = {np.round(Qs[1612-minI],2)}',cbarShow=False)

# %%
