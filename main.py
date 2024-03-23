#%%
from defineCrystal import TopoCav
from inverseDesign import of_Q
from saveLoad import experiment
from inverseDesign import ID
import os
import numpy as np
import legume
from process import fieldPlotS3
# Set to the number of CPU cores you want to use, force max core useage
os.environ['MKL_NUM_THREADS'] = '36'
os.environ['MKL_DYNAMIC'] = 'FALSE'
#%%

#check the S3 plot for the different points of interest
pi = np.pi
t = np.sqrt(3)
kpoints = np.array([[2*np.pi/t,-2*np.pi/t,-pi/t,pi/t,-pi/t,pi/t],
                    [0,0,-pi,pi,pi,-pi]])

index = 1960
options = {'verbose': False, 'gradients': 'approx',
                           'numeig': index+1,
                           'compute_im': False
                        }

phc, lattice = TopoCav(sideLength=21,Nx=44,Ny=44)
gme = legume.GuidedModeExp(phc, gmax=1.38157894736)
gme.run(kpoints=kpoints, **options)

plot_path = 'results/S3'
os.makedirs(plot_path, exist_ok=True)
for i in range(kpoints[0].size):
      fieldPlotS3(phc,gme,gapIndex=index,resolution=400,title=f'kx={np.round(kpoints[0,i],2)}, ky = {np.round(kpoints[1,i],2)}',cbarShow=True,save=True,path=plot_path+f'/{i}')



#%%

#optomization that we will run

dr = {(10,11,1):0,(10,10,0):0,(10,10,1):0,(9,11,1):0,(9,10,0):0,(9,9,0):0,
      (-11,11,1):0,(-10,11,1):0,(-10,10,0):0,(-10,10,1):0,(-9,10,0):0,(-10,9,0):0,
      (0,-10,0):0,(0,-10,1):0,(0,-9,0):0,(0,-9,1):0,(-1,-9,0):0,(-1,-9,1):0}

runs = {'name':'topoTest',
        'lbfgsRads':{'dr':dr,'method':'l-bfgs-b','crystal':TopoCav,'sideLength':21,
                     'gmax':1.38157894736,'Nx':44,'Ny':44,'objective_function':of_Q,'optMode':1960,
                     'dslab':170/266,'n_slab':11.6,'ra':125/(2*266),'ra1':56/(266*2)}}

experiment(runs,ID)

#%%

#this is the large optomization of the L3 cavity
#now run optomization 
# from saveLoad import experiment
# from inverseDesign import ID
# from genConst import L3const
# import os


# dx = {(2,0):0,(3,0):0,(4,0):0,(-2,0):0,(-3,0):0,(-4,0):0,
#       (0,1):0,(1,1):0,(2,1):0,(-1,1):0,(-2,1):0,(-3,1):0,
#       (0,-1):0,(1,-1):0,(2,-1):0,(-1,-1):0,(-2,-1):0,(-3,-1):0}

# dy = {(2,0):0,(3,0):0,(4,0):0,(-2,0):0,(-3,0):0,(-4,0):0,
#       (0,1):0,(1,1):0,(2,1):0,(-1,1):0,(-2,1):0,(-3,1):0,
#       (0,-1):0,(1,-1):0,(2,-1):0,(-1,-1):0,(-2,-1):0,(-3,-1):0}

# dr = {(2,0):0,(3,0):0,(4,0):0,(-2,0):0,(-3,0):0,(-4,0):0,
#       (0,1):0,(1,1):0,(2,1):0,(-1,1):0,(-2,1):0,(-3,1):0,
#       (0,-1):0,(1,-1):0,(2,-1):0,(-1,-1):0,(-2,-1):0,(-3,-1):0}

# runs = {'name':'freqConfine',
#         #'lbfgsbBig': {'dx':dx,'dy':dy,'dr':dr,'method':'l-bfgs-b'},
#         'trust-constrBig': {'dx':dx,'dy':dy,'dr':dr,'method':'trust-constr','constraints':True,'minrad':.05,'mindist':.05,
#                          'minfreq':.261,'maxfreq':.3,'constFunc':L3const}}

# experiment(runs,ID)



# %%
