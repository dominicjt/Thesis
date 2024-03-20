#%%
from defineCrystal import TopoCav
from inverseDesign import of_QV
from saveLoad import experiment
from inverseDesign import ID
import os
# Set to the number of CPU cores you want to use, force max core useage
os.environ['MKL_NUM_THREADS'] = '36'
os.environ['MKL_DYNAMIC'] = 'FALSE'
#%%

#check the S3 plot for the different points of interest



#%%

#optomization that we will run

dr = {(10,11,1):0,(10,10,0):0,(10,10,1):0,(9,11,1):0,(9,10,0):0,(9,9,0):0,
      (-11,11,1):0,(-10,11,1):0,(-10,10,0):0,(-10,10,1):0,(-9,10,0):0,(-10,9,0):0,
      (0,-10,0):0,(0,-10,1):0,(0,-9,0):0,(0,-9,1):0,(-1,-9,0):0,(-1,-9,1):0}

runs = {'name':'topoTest',
        'lbfgsRads':{'dr':dr,'method':'l-bfgs-b','crystal':TopoCav,'sideLength':21,
                     'gmax':1.38157894736,'Nx':44,'Ny':44,'objective_function':of_QV,'optMode':1960,
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
