#%%
from defineCrystal import TopoCav
import legume 
import numpy as np
import time 
from process import fieldPlot, fieldPlotS3
import json
import os
#%%
# Set to the number of CPU cores you want to use, force max core useage
os.environ['MKL_NUM_THREADS'] = '36'
os.environ['MKL_DYNAMIC'] = 'FALSE'


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

#extracts frist argument from compute rad for parallelizing
def compute_rad_p(point):
    try:
        iter(point)
    except TypeError:
        point = [point]
    (freq_im, _, _) = gme.compute_rad(0, point)
    return freq_im[0]

#plotter wrapper for parallizing
def plotter_wrapper(call):
    fieldPlot(*call['args'],**call['kwargs'])

# %%
#set up loop constants
size = [44]
gmaxs = np.linspace(1.75,2,10)[1:]
cavity = [21]

file_path = 'results/gmaxTest/gmaxTest.json'

# Loop through each array
for i, s in enumerate(size):
    for j, g in enumerate(gmaxs):
        for k, c in enumerate(cavity):

            options = {'verbose': False, 'gradients': 'approx',
                       'numeig': s*s+200,       # get 5 eigenvalues
                       'compute_im': False
                       }
            
            #run simulation
            t = time.time()
            phc, lattice = TopoCav(sideLength=c,Nx=s,Ny=s)
            gme = legume.GuidedModeExp(phc, gmax=g)
            gme.run(kpoints=np.array([[0],[0]]), **options)
            t_tot = time.time()-t
            
            NTindices = np.where((266/gme.freqs[0] > 1015) & (266/gme.freqs[0] < 1028))[0]
            minI = NTindices[0]
            maxI = NTindices[-1]+1

            #attempting multithredding is to memory intensive
            # Using ThreadPoolExecutor to parallelize the computation
            #with concurrent.futures.ThreadPoolExecutor() as executor:
                # Map the function over the points
            #    freq_im = list(executor.map(compute_rad_p, NTindices))
            freq_im = np.array(freq_im)

            (freq_im, _, _) = gme.compute_rad(0,NTindices)

            datatosave = {'time':t_tot,'gmax':g,'cavity':c,'size':s,
                        'NTindices':NTindices.tolist(),'NTwavelength':(266/gme.freqs[0,minI:maxI]).tolist(),
                        'Q':(gme.freqs[0,minI:maxI]/(2*freq_im)).tolist()}
            
            data = read_data(file_path)
            data.append(datatosave)
            write_data(file_path, data)

            #generate a list of dictionaries fro the arguments
            #calls = []
            #for i in NTindices:
            #    args = {'args': [phc,gme],'kwargs': {'gapIndex': i,'resolution':200,'title':f'Wavelength = {np.round(266/gme.freqs[0,i],2)}, size={s}\ncavity={c}, gmax={g}',
            #                                          'cbarShow':False,'save':True,'path':f'results/convTest/cavity{s}{c}{int(g*100)}/cavity{i}.png'}}
            #    calls.append(args)

            #save plots
            os.makedirs(f'results/gmaxTest/cavity{s}{c}{int(g*100)}', exist_ok=True)
            for i in NTindices:
                fieldPlot(phc,gme,gapIndex=i,resolution=200,title=f'Wavelength = {np.round(266/gme.freqs[0,i],2)}, size={s}\ncavity={c}, gmax={np.round(g,2)}',cbarShow=False,save=True,path=f'results/gmaxTest/cavity{s}{c}{int(g*100)}/{i}')











#%%
#now run optomization 
from saveLoad import experiment
from inverseDesign import ID
from genConst import L3const
import os


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

experiment(runs,ID)


