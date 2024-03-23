#%%
from defineCrystal import TopoCav, LNCrystal
import legume 
import numpy as np
import time 
from process import fieldPlot, fieldPlotS3, crystalPlot
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.image as mpimg
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
gmaxs = [1]
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
            
            NTindices = np.where((266/gme.freqs[0] > 1021) & (266/gme.freqs[0] < 1028))[0]
            minI = NTindices[0]
            maxI = NTindices[-1]+1

            #attempting multithredding is to memory intensive
            # Using ThreadPoolExecutor to parallelize the computation
            #with concurrent.futures.ThreadPoolExecutor() as executor:
                # Map the function over the points
            #    freq_im = list(executor.map(compute_rad_p, NTindices))
            #freq_im = np.array(freq_im)

            #(freq_im, _, _) = gme.compute_rad(0,NTindices)

            #datatosave = {'time':t_tot,'gmax':g,'cavity':c,'size':s,
            #            'NTindices':NTindices.tolist(),'NTwavelength':(266/gme.freqs[0,minI:maxI]).tolist(),
            #            'Q':(gme.freqs[0,minI:maxI]/(2*freq_im)).tolist()}
            
            #data = read_data(file_path)
            #data.append(datatosave)
            #write_data(file_path, data)

            #generate a list of dictionaries fro the arguments
            #calls = []
            #for i in NTindices:
            #    args = {'args': [phc,gme],'kwargs': {'gapIndex': i,'resolution':200,'title':f'Wavelength = {np.round(266/gme.freqs[0,i],2)}, size={s}\ncavity={c}, gmax={g}',
            #                                          'cbarShow':False,'save':True,'path':f'results/convTest/cavity{s}{c}{int(g*100)}/cavity{i}.png'}}
            #    calls.append(args)

            #save plots
            os.makedirs(f'results/gmaxTest/cavity{s}{c}{int(g*100)}', exist_ok=True)
            for i in NTindices:
                fieldPlotS3(phc,gme,gapIndex=i,resolution=200,title=f'Wavelength = {np.round(266/gme.freqs[0,i],2)}, size={s}\ncavity={c}, gmax={g}',cbarShow=False,save=True,path=f'results/gmaxTest/cavity{s}{c}{int(g*100)}/S3{i}')

            

# %%
data2 = read_data("results/convTest/dataSecond.json")
data1 = read_data("results/convTest/dataFirst.json")
data3 = read_data("results/convTest/data.json")
dataold = np.array(data1+data2+data3)
data = np.array(read_data("results/gmaxTest/gmaxTest.json"))

#%%

# Define the colormap
cmap = cm.Greys

# Create a Normalize object for the colorbar. This will scale 10^4 to 10^6
norm = mcolors.LogNorm(vmin=1e4, vmax=1e6)

plt.rcParams.update({'font.size': 16})
size = 44
cavity = 21
for d in data:
    if d['size']==44 and d['cavity']==cavity:
        
        #plot get the quality factors set up for colloring
        Qs = d['Q']

        scatter = plt.scatter(d['NTwavelength'],np.ones_like(d['NTwavelength'])*d["gmax"],
                 c=Qs, cmap='Greys', edgecolors='black', linewidth=1.5,norm=norm)  

cbar = plt.colorbar(scatter, format='${%d}$')  
cbar.set_ticks([1e4, 1e5, 1e6])  # Setting ticks for the colorbar
cbar.set_ticklabels(['$10^4$', '$10^5$', '$10^6$'])  # Setting custom tick labels
plt.xlabel('Wavelength [nm]')
plt.ylabel('gmax')
plt.title(f"size = {size}, cavity = {cavity}")
plt.show()
# %%
for d in data:
    if d['gmax']==1.75:
        plt.scatter(d['NTwavelength'],d['Q'])
plt.xlim(1015,1029)
plt.show()
#%%
for d in data:
    plt.scatter(d['gmax'],d['time'])
plt.show()
# %%
data2 = read_data("results/freqConfine/trust-constrBig.json")['iterations']
for key,value in data2.items():
    plt.scatter(int(key),-value['value'])
    #plt.scatter(int(key),value['freq'])
plt.yscale('log')
plt.show()
# %%
import ast
dx1 = data2['205']['dx']
dy1 = data2['205']['dy']
dr1 = data2['205']['dr']

dx={};dy={};dr={}
for key,value in dx1.items():
    dx[ast.literal_eval(key)] = value

for key,value in dy1.items():
    dy[ast.literal_eval(key)] = value

for key,value in dr1.items():
    dr[ast.literal_eval(key)] = value

phc = LNCrystal(Nx=16,Ny=10,dslab=.6,n_slab=12,ra=.29,dx=dx,dy=dy,dr=dr)
legume.viz.eps_xy(phc)
optoins = {'verbose':False,'numeig':161,'compute_im':False}
gme = legume.GuidedModeExp(phc,gmax=2)
gme.run(kpoints=np.array([[0],[0]]),**options)
(freq_im,_,_) = gme.compute_rad(0,[160])
# %%
fieldPlot(phc,gme,gapIndex=160,resolution=200,cbarShow=False)
# %%
print('Q=',gme.freqs[0,160]/(2*freq_im[0]))
# %%
for d in dataold:
    if d["gmax"]==1.5 and d["cavity"]==21:
        plt.scatter(d["NTwavelength"],np.ones_like(d["NTwavelength"])*d["size"])
plt.xlim(1021,1028)
plt.show()
# %%
