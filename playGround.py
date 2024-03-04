#%%
from defineCrystal import LNCrystal, TopoCav
import legume 
import numpy as np
import time 
import matplotlib.pyplot as plt
from process import fieldPlot
import json
import os
#%%

# Function to read existing data from JSON file
def read_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Function to write data to JSON file
def write_data(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# %%
#set up loop constants
size = np.arange(40,80,2)
gmaxs = [1,1.25,1.5,1.75,2]
cavity = [9,15,21]

file_path = 'results/convTest/data.json'

# Ensure the file is cleared if it exists when the code runs
if os.path.exists(file_path):
    open(file_path, 'w').close()

# Loop through each array
for i, s in enumerate(size):
    for j, g in enumerate(gmaxs):
        for k, c in enumerate(cavity):
            print(s,g,c)

            options = {'verbose': False, 'gradients': 'approx',
                       'numeig': s*s+200,       # get 5 eigenvalues
                       'compute_im': False
                       }
            
            #run simulation
            t = time.time()
            phc,_ = TopoCav(sideLength=c,Nx=s,Ny=s)
            gme = legume.GuidedModeExp(phc, gmax=g)
            gme.run(kpoints=np.array([[0],[0]]), **options)

            #compute values of interest
            NTindices = np.where((266/gme.freqs[0] > 1000) & (266/gme.freqs[0] < 1035))[0]
            minI = NTindices[0]
            maxI = NTindices[-1]+1
            (freq_im, _, _) = gme.compute_rad(0, NTindices)
            t_tot = time.time()-t

            #save data
            existing_data = read_data(file_path)
            new_data = {'time':t_tot, 'size':int(s), 'gmax':g, 'cavity':c, 'NTindices':NTindices.tolist(),
                    'NTwavelength': (266/gme.freqs[0,minI:maxI]).tolist(), 'Q': (gme.freqs[0,minI:maxI]/(2*freq_im)).tolist()}
            existing_data.append(new_data)
            write_data(file_path, existing_data)

            #save plots
            os.makedirs(f'results/convTest/cavity{s}{c}{g*100}', exist_ok=True)
            for i in NTindices:
                fieldPlot(phc,gme,gapIndex=i,resolution=200,title=f'Wavelength = {np.round(266/gme.freqs[0,i],2)}, size={s}\ncavity={c}, gmax={g}',cbarShow=False,save=True,path=f'results/convTest/cavity{s}{c}{g*100}/cavity{i}.png')


            
            

# %%
