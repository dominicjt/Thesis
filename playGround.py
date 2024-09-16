#%%
from defineCrystal import TopoWave
import legume 
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import norm
# %%
with open('results/noise/inital/t4.json') as file:
    data = np.array(json.load(file))

#plt.hist(data[:,0],bins=50,edgecolor='black')
plt.vlines([48156629,5875],0,1,'r',linestyle='--')
plt.xscale('log')

xs = np.logspace(np.log10(5875)-.5,np.log10(48156629)+.5,1000)
linear = np.log10(data[:,0])
mean = np.mean(linear)
std = np.std(linear)
gaus = norm.pdf(np.log10(xs),loc=mean,scale=std)
gaus = gaus/np.max(gaus)
plt.plot(xs,gaus)
plt.xscale('log')
plt.show()
print(std)
print(mean)

# %%
with open('results/noise/topoConv/t1.json')as file:
    data = json.load(file)
s=200
plt.scatter(np.arange(0,60),gme2.freqs[0][1600:1660],s=s)
for arr in data:
    s=s/1.5
    plt.scatter(np.arange(0,60),arr[1600:1660],s=s)
    print('ran')
plt.show()

# %%

# %%
