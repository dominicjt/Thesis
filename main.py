#%%
from defineCrystal import TopoCav
from inverseDesign import of_Q, of_QV
from saveLoad import experiment
from inverseDesign import ID
import os
import numpy as np
import legume
from process import fieldPlot, convergance
# Set to the number of CPU cores you want to use, force max core useage
os.environ['MKL_NUM_THREADS'] = '36'
os.environ['MKL_DYNAMIC'] = 'FALSE'
#%%
# size = [24]
# gmaxs = np.linspace(1,4,22)
# cavity = [7,9,11]
# guidedModes = [[0]]

# convergance(size,gmaxs,cavity,guidedModes,'results/triConv2',266/1040,266/980)


#%%
#check the S3 plot for the different points of interest
# pi = np.pi
# t = np.sqrt(3)
# kpoints = np.array([[2*np.pi/t,-2*np.pi/t,-pi/t,pi/t,-pi/t,pi/t],
#                     [0,0,-pi,pi,pi,-pi]])

# index = 1960
# options = {'verbose': False, 'gradients': 'approx',
#                            'numeig': index+1,
#                            'compute_im': False
#                         }

# phc, lattice = TopoCav(sideLength=21,Nx=44,Ny=44)
# gme = legume.GuidedModeExp(phc, gmax=1.38157894736)
# gme.run(kpoints=kpoints, **options)

# plot_path = 'results/S3'
# os.makedirs(plot_path, exist_ok=True)
# for i in range(kpoints[0].size):
#       fieldPlotS3(phc,gme,gapIndex=index,resolution=400,title=f'kx={np.round(kpoints[0,i],2)}, ky = {np.round(kpoints[1,i],2)}',cbarShow=True,save=True,path=plot_path+f'/{i}')



# #%%

# #optomization that we will run

# dr = {(10,11,1):0,(10,10,0):0,(10,10,1):0,(9,11,1):0,(9,10,0):0,(9,9,0):0,
#       (-11,11,1):0,(-10,11,1):0,(-10,10,0):0,(-10,10,1):0,(-9,10,0):0,(-10,9,0):0,
#       (0,-10,0):0,(0,-10,1):0,(0,-9,0):0,(0,-9,1):0,(-1,-9,0):0,(-1,-9,1):0}

# runs = {'name':'topoTest',
#         'lbfgsRads':{'dr':dr,'method':'l-bfgs-b','crystal':TopoCav,'sideLength':21,
#                      'gmax':1.38157894736,'Nx':44,'Ny':44,'objective_function':of_Q,'optMode':1960,
#                      'dslab':170/266,'n_slab':11.6,'ra':125/(2*266),'ra1':56/(266*2)}}

# experiment(runs,ID)

#%%

#this is the large optomization of the L3 cavity
#now run optomization 
from saveLoad import experiment
from inverseDesign import ID
from genConst import TopoWavconstSingleMode, W1const
from defineCrystal import TopoWave, W1
import os
import numpy as np 
import json
from inverseDesign import of_alphaDng
os.environ['MKL_NUM_THREADS'] = '36'
os.environ['MKL_DYNAMIC'] = 'FALSE'

#toni's crystal
a = 403 # nm, the lattice constant (pitch) of the crystal
h = 270 # nm, the height (depth) of the slab
r0, r1 = .105,.235  # nm, starting radii

nk = 25 # Number of k-points
kmin, kmax = np.pi*.5, np.pi # Min and max k-values
path = np.vstack((np.linspace(kmin, kmax, nk), np.zeros(nk))) #k vectors
kpoints = (tuple(path[:,8]),(tuple(path[:,4]),tuple(path[:,6]),),(tuple(path[:,12]),tuple(path[:,16]),tuple(path[:,20]),))

dxW1 = {(0,0):0,(0,1):0,(0,2):0,(0,3):0,(0,-1):0,(0,-2):0,(0,-3):0}

dyW1 = {(0,0):0,(0,1):0,(0,2):0,(0,3):0,(0,-1):0,(0,-2):0,(0,-3):0}

drW1 = {(0,0):0,(0,1):0,(0,2):0,(0,3):0,(0,-1):0,(0,-2):0,(0,-3):0}

with open('results/ForceParamTest/51000.1.json') as file:
    data = json.load(file)['iterations']

last = list(data.values())[509]

dxF = {eval(key):value for key,value in last['dx'].items()}
dyF = {eval(key):value for key,value in last['dy'].items()}
drF = {eval(key):value for key,value in last['dr'].items()}

dx = {**dxF,**{(0,-2,0):0,(0,-2,1):0,(0,3,0):0,(0,3,1):0}}
dy = {**dyF,**{(0,-2,0):0,(0,-2,1):0,(0,3,0):0,(0,3,1):0}}
dr = {**drF,**{(0,-2,0):0,(0,-2,1):0,(0,3,0):0,(0,3,1):0}}

runs = {'name':'alphaDng_runs2',
                  'W1Test':{'dx': dxW1, 'dy': dyW1, 'dr': drW1,'Nx': 1, 'Ny': 20, 'dslab': 170/266, 'n_slab': 3.4638**2,'ra': .3,
                        'gmax': 2, 'method': 'trust-constr', 'objective_function': of_alphaDng, 'nk':1,
                        'bounds': None, 'gradients': 'exact', 'compute_im': False, 'callback': None,
                        'constraints':True,'constFunc':W1const,'minfreq':.27,'maxfreq':.28,'minrad':27.5/266,'mindist':40/266,"optMode":20,
                        'crystal':W1,'kpoints':kpoints,'a':266,"bandwidth":.005}}
                #   'BIWTest3': {'dx': dx, 'dy': dy, 'dr': dr,'Nx': 1, 'Ny': 21, 'dslab': 170/266, 'n_slab': 3.4638**2,'ra': (r0,r1),
                #         'gmax': 2, 'method': 'trust-constr', 'objective_function': of_alphaDng, 'nk':1,
                #         'bounds': None, 'gradients': 'exact', 'compute_im': False, 'callback': None,
                #         'constraints':True,'constFunc':TopoWavconstSingleMode,'minfreq':.260,'maxfreq':.275,'minrad':27.5/266,'mindist':40/266,"optMode":20,
                #         'crystal':TopoWave,'kpoints':kpoints,'a':266,"bandwidth":.005}}

experiment(runs,ID)

# with open('results/ForceParamTest/51000.1.json') as file:
#     data = json.load(file)['iterations']

# last = list(data.values())[200]

# dxF = {eval(key):value for key,value in last['dx'].items()}
# dyF = {eval(key):value for key,value in last['dy'].items()}
# drF = {eval(key):value for key,value in last['dr'].items()}

# dx = {**dxF,**{(0,-2,0):0,(0,-2,1):0,(0,3,0):0,(0,3,1):0}}
# dy = {**dyF,**{(0,-2,0):0,(0,-2,1):0,(0,3,0):0,(0,3,1):0}}
# dr = {**drF,**{(0,-2,0):0,(0,-2,1):0,(0,3,0):0,(0,3,1):0}}

# runs = {'name':'alphaDng_runs',
#                     'BIW_BW200': {'dx': dx, 'dy': dy, 'dr': dr,'Nx': 1, 'Ny': 21, 'dslab': 170/266, 'n_slab': 3.4638**2,'ra': (r0,r1),
#                         'gmax': 2, 'method': 'trust-constr', 'objective_function': of_alphaDng, 'nk':1,
#                         'bounds': None, 'gradients': 'exact', 'compute_im': False, 'callback': None,
#                         'constraints':True,'constFunc':TopoWavconstSingleMode,'minfreq':.255,'maxfreq':.27,'minrad':27.5/266,'mindist':40/266,"optMode":20,
#                         'crystal':TopoWave,'kpoints':(tuple(path[:,8]),(tuple(path[:,4]),),(tuple(path[:,18]),tuple(path[:,24]),)),'a':266,"bandwidth":.005},
#                     'BIW_BW250_larger': {'dx': dx, 'dy': dy, 'dr': dr,'Nx': 1, 'Ny': 21, 'dslab': 170/266, 'n_slab': 3.4638**2,'ra': (r0,r1),
#                         'gmax': 2, 'method': 'trust-constr', 'objective_function': of_alphaDng, 'nk':1,
#                         'bounds': None, 'gradients': 'exact', 'compute_im': False, 'callback': None,
#                         'constraints':True,'constFunc':TopoWavconstSingleMode,'minfreq':.265,'maxfreq':.275,'minrad':27.5/266,'mindist':40/266,"optMode":20,
#                         'crystal':TopoWave,'kpoints':(tuple(path[:,8]),(tuple(path[:,4]),),(tuple(path[:,18]),tuple(path[:,24]),)),'a':266,"bandwidth":.008}}

# experiment(runs,ID)

# with open('results/ForceParamTest/51000.1.json') as file:
#     data = json.load(file)['iterations']

# last = list(data.values())[250]

# dxF = {eval(key):value for key,value in last['dx'].items()}
# dyF = {eval(key):value for key,value in last['dy'].items()}
# drF = {eval(key):value for key,value in last['dr'].items()}

# dx = {**dxF,**{(0,-2,0):0,(0,-2,1):0,(0,3,0):0,(0,3,1):0}}
# dy = {**dyF,**{(0,-2,0):0,(0,-2,1):0,(0,3,0):0,(0,3,1):0}}
# dr = {**drF,**{(0,-2,0):0,(0,-2,1):0,(0,3,0):0,(0,3,1):0}}

# runs = {'name':'alphaDng_runs',
#                     'BIW_BW250': {'dx': dx, 'dy': dy, 'dr': dr,'Nx': 1, 'Ny': 21, 'dslab': 170/266, 'n_slab': 3.4638**2,'ra': (r0,r1),
#                         'gmax': 2, 'method': 'trust-constr', 'objective_function': of_alphaDng, 'nk':1,
#                         'bounds': None, 'gradients': 'exact', 'compute_im': False, 'callback': None,
#                         'constraints':True,'constFunc':TopoWavconstSingleMode,'minfreq':.255,'maxfreq':.27,'minrad':27.5/266,'mindist':40/266,"optMode":20,
#                         'crystal':TopoWave,'kpoints':(tuple(path[:,8]),(tuple(path[:,4]),),(tuple(path[:,18]),tuple(path[:,24]),)),'a':266,"bandwidth":.005},
#                     'BIW_BW250_larger': {'dx': dx, 'dy': dy, 'dr': dr,'Nx': 1, 'Ny': 21, 'dslab': 170/266, 'n_slab': 3.4638**2,'ra': (r0,r1),
#                         'gmax': 2, 'method': 'trust-constr', 'objective_function': of_alphaDng, 'nk':1,
#                         'bounds': None, 'gradients': 'exact', 'compute_im': False, 'callback': None,
#                         'constraints':True,'constFunc':TopoWavconstSingleMode,'minfreq':.265,'maxfreq':.275,'minrad':27.5/266,'mindist':40/266,"optMode":20,
#                         'crystal':TopoWave,'kpoints':(tuple(path[:,8]),(tuple(path[:,4]),),(tuple(path[:,18]),tuple(path[:,24]),)),'a':266,"bandwidth":.008}}

# experiment(runs,ID)


# %%
