#%%
from saveLoad import experiment
from inverseDesign import ID
from genConst import L3const
import numpy as np


dx = {(2,0):0,(-2,0):0}
consts = L3const(minrad=.05,mindist=.05,dx=dx)

runs = {'name':'test',
        'trust-constr': {'dx':dx,'method':'trust-constr','constraints':consts}}

experiment(runs,ID)


#%%