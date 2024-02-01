#%%
from saveLoad import experiment
from inverseDesign import ID

runs = {'name':'test',
        'trust-constr': {'dx':{(2,0):0,(-2,0):0},'model':'trust-constr'}}

experiment(runs,ID)


# %%
