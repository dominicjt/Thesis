#%%
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import legume
from defineCrystal import TopoWave,W1
from IPython.display import HTML
from matplotlib.patches import Circle
from inverseDesign import calc_purcellEnhance
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
# %%
with open('results/alphaDng_runs2/BIWTest3.json') as file:
    out = json.load(file)
data = out['iterations']
metadata = out['metadata']

#otherloss = np.array([run['other'][1] for run in data.values()])
vals = np.array([run['value'] for run in data.values()])
freqs = np.array([run['freq'][0][0] for run in data.values()])
freqs_all = np.array([run['freq'][0][2] for run in data.values()])

cmap = plt.get_cmap('viridis')
normalize = plt.Normalize(vmin=0,vmax=len(vals))
colors = cmap(normalize(np.arange(len(vals))))
plt.scatter(freqs,-vals,c=colors,s=100)
plt.colorbar(plt.cm.ScalarMappable(norm=normalize,cmap=cmap),label='Epoch')
plt.axvline(x=metadata['minfreq'],color='r',linestyle='--')
plt.axvline(x=metadata['maxfreq'],color='r',linestyle='--')
plt.xlabel('frequency')
plt.ylabel('alpha')
plt.show()

cmap = plt.get_cmap('gist_rainbow')
normalize = plt.Normalize(vmin=np.min(-vals),vmax=np.max(-vals))
colors = cmap(normalize(-vals))
sizes = np.linspace(1,.5,len(vals))*100
plt.scatter(freqs,np.min(freqs_all,axis=1),c=colors,s=sizes,edgecolor='k')
plt.colorbar(plt.cm.ScalarMappable(norm=normalize,cmap=cmap),label='Epoch')
xd = np.linspace(metadata['minfreq'],metadata['maxfreq'],2)
plt.plot(xd,xd,'--r')
plt.plot([metadata['minfreq'],metadata['minfreq']],[metadata['minfreq'],.3],'--r')
plt.plot([metadata['maxfreq'],metadata['maxfreq']],[metadata['maxfreq'],.3],'--r')
plt.xlabel('frequency')
plt.ylabel('alpha')
plt.ylim(.2,.3)
# plt.ylim(.280,.3)
# plt.xlim(.280,.3)
plt.show()
plt.plot(-vals,'o')

# %%
#toni's crystal
a = metadata['a'] # nm, the lattice constant (pitch) of the crystal
h = metadata['dslab']*a # nm, the height (depth) of the slab
ra = metadata['ra']  # nm, starting radii

nk = 25 # Number of k-points
kmin, kmax = np.pi*.5, np.pi # Min and max k-values
path = np.vstack((np.linspace(kmin, kmax, nk), np.zeros(nk))) #k vectors

last = list(data.values())[-1]

dx = {eval(key):value for key,value in last['dx'].items()}
dy = {eval(key):value for key,value in last['dy'].items()}
dr = {eval(key):value for key,value in last['dr'].items()}
#%%
phc,_ = globals()[metadata["crystal"]](Ny=metadata['Ny'],n_slab=metadata['n_slab'],ra=ra,dslab=h/a,dx=dx,dy=dy,dr=dr)
out = legume.viz.eps_xy(phc,Nx=40,Ny=400)
#%%
gme = legume.GuidedModeExp(phc,gmax=2)
gme.run(kpoints=path,numeig=50,compute_im=False)
#%%
xs = np.linspace(kmin,kmax,nk)/(2*np.pi)
plt.plot(xs,gme.freqs[:,19:22])
plt.axvline(metadata['kpoints'][0][0]/2/np.pi,color='red')
for i in range(len(metadata['kpoints'][1])):
    plt.axvline(metadata['kpoints'][1][i][0]/2/np.pi)
for i in range(len(metadata['kpoints'][2])):
    plt.axvline(metadata['kpoints'][2][i][0]/2/np.pi)
plt.axvline(path[0][14]/2/np.pi)
plt.axhline(metadata['maxfreq'],color='r',linestyle='--')
plt.axhline(metadata['minfreq'],color='r',linestyle='--')
plt.xlabel('K')
plt.ylabel('Frequency')
plt.title('Without')
plt.show()

#%%
plt.rcParams.update({'font.size':12})
ylim = 10*np.sqrt(3)/2
ys = np.linspace(-ylim,ylim,200)
fields,_,_ = gme.get_field_xy('E',8,20,phc.layers[0].d/2,ygrid=ys,component='xyz')
eabs = np.abs(np.conj(fields['x'])*fields['x']+np.conj(fields['y'])*fields['y']+np.conj(fields['z'])*fields['z'])
plt.imshow(eabs)


#%%
#takes in the data for that itteration and returns the phc and gme for that frame
def itterSim(data,k=8,rungme=True):

    #pull out the relevint informatoin
    dx = {eval(key):value for key,value in data['dx'].items()}
    dy = {eval(key):value for key,value in data['dy'].items()}
    dr = {eval(key):value for key,value in data['dr'].items()}

    #generates pathe
    nk = 25 # Number of k-points
    kmin, kmax = np.pi*.5, np.pi # Min and max k-values
    path = np.vstack((np.linspace(kmin, kmax, nk), np.zeros(nk))) #k vectors
    path2 = np.array([[path[0][k],path[0][k]+.001],[path[1][k],path[1][k]]])
    
    #in the future we shoudl make these values inputs, but this makes phc
    phc,_ = TopoWave(Ny=metadata['Ny'],n_slab=metadata['n_slab'],ra=ra,dslab=h/a,dx=dx,dy=dy,dr=dr)
    if not rungme:
        return(phc)
    #runs simulatoin
    gme = legume.GuidedModeExp(phc,gmax=2)

    gme2 = legume.GuidedModeExp(phc,gmax=2)
    gme.run(kpoints=path,numeig=30,compute_im=False,verbose=False)
    gme2.run(kpoints=path2,numeig=30,compute_im=False,verbose=False)

    return(gme,gme2,phc)


#this should return the chiral field for each frame
def chiralField(gme,phc):
    fp,fn,mask = calc_purcellEnhance(gme,phc,20,0,403,Nx=60,Ny=300)
    forwardMasked = mask*(fp*((fp-fn)/(fn+fp)))
    return(forwardMasked)



#%%
plt.rcParams.update({'font.size':16})
every = 50
# Create a figure with custom subplots
fig = plt.figure(figsize=(8, 8))

# Axes for the top line plot
ax1 = fig.add_axes([0.1, 0.75, 0.8, 0.2])  # [left, bottom, width, height]
line, = ax1.plot([0], -vals[0], 'o')
#ax1.set_yscale('log')
ax1.set_ylim(np.min(-vals)-.2, np.max(-vals)+.2)
ax1.set_xlim(0, len(vals))
ax1.set_xlabel('Itteration')
ax1.set_ylabel(r'$-\log(\alpha)$')

# Axes for the imshow structure plot
ax2 = fig.add_axes([0.1, 0.1, 0.115, 3*np.sqrt(3)*.105])  # [left, bottom, width, height]
structure_data = np.random.rand(20, 100)*.25  # Dummy data for structure
structure_plot = ax2.imshow(structure_data, extent=[-.5,.5,-3*np.sqrt(3)/2,3*np.sqrt(3)/2], cmap='cool' ,aspect='auto',origin='lower')
#levels = np.array([-1e5,-1e4,-1e3,-1e2,-1e1,-1,0,1,1e1,1e2,1e3,1e4,1e5])
#norm = BoundaryNorm(boundaries=levels,ncolors=256,clip=False)
#structure_plot.set_norm(norm)
#fig.colorbar(structure_plot,ax=ax2)
ax2.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
ax2.set_title(r'$|E|^2$')

#get circles to plot
phc = itterSim(data[str(0)],rungme=False)
circles = [Circle((s.x_cent,s.y_cent),s.r,edgecolor='black',facecolor='none',linewidth=4) for s in phc.layers[0].shapes]
for c in circles:
    ax2.add_patch(c)
circlesAround = [Circle((0,0),edgecolor='black',facecolor='none',linewidth=4) for _ in range(len(circles))]
for i,ca in enumerate(circlesAround):
    ca.center = (circles[i].center[0]-np.sign(circles[i].center[0]),circles[i].center[1])
    ca.radius = circles[i].radius
    ax2.add_patch(ca)

# Axes for the line plot with shaded regions
ax3 = fig.add_axes([.35, 0.1, 0.55, 3*np.sqrt(3)*.105])  # [left, bottom, width, height]
line1, = ax3.plot([], [], linewidth=2)
line2, = ax3.plot([], [], linewidth=2)
shade1 = None  # Placeholder for shaded region above line1
shade2 = None  # Placeholder for shaded region below line2
kpoint, = ax3.plot([],[],'o',markersize=10,color='red',markeredgecolor='black')
ax3.set_xlim(.25, .5)
ax3.set_ylim(.25, .33)
ax3.set_ylabel('Frequency')
ax3.set_xlabel('k')

# Initialize function to set the plots
def init():
    line.set_data([], [])
    structure_plot.set_data(np.random.rand(20, 100))  # Initial data for structure
    phc = itterSim(data[str(0)],rungme=False)
    for i,c in enumerate(circles):
        c.center = (phc.layers[0].shapes[i].x_cent,phc.layers[0].shapes[i].y_cent)
        c.radius = phc.layers[0].shapes[i].r
    for i,ca in enumerate(circlesAround):
        x = phc.layers[0].shapes[i].x_cent
        ca.center = (x-np.sign(x),phc.layers[0].shapes[i].y_cent)
        ca.radius = phc.layers[0].shapes[i].r
    line1.set_data([], [])
    line2.set_data([], [])
    kpoint.set_data([],[])
    global shade1, shade2
    shade1 = ax3.fill_between([], [], [], color='gray',edgecolor='black', alpha=0.5)  # Initial shaded region
    shade2 = ax3.fill_between([], [], [], color='gray',edgecolor='black', alpha=0.5)  # Initial shaded region
    return line, structure_plot, line1, line2, shade1, shade2, kpoint, *circles, *circlesAround

# Update function for each frame
def update(frame):
    print(frame)
    frame = frame*every
    # Update the top line plot
    x = np.arange(0, frame)
    y = -vals[:frame]
    line.set_data(x, y)

    # Update the structure plot with static data (or modify if needed)
    ylim = 5*np.sqrt(3)/2
    ys = np.linspace(-ylim,ylim,200)
    gme,gme2,phc = itterSim(data[str(frame)])
    #structure_data = chiralField(gme2,phc)
    fields,_,_ = gme.get_field_xy('E',8,20,phc.layers[0].d/2,ygrid=ys,Nx=40,component='xyz')
    structure_plot.set_data(np.abs(np.conj(fields['x'])*fields['x']+np.conj(fields['y'])*fields['y']+np.conj(fields['z'])*fields['z']))
    #structure_plot.set_norm(TwoSlopeNorm(vmin=np.min(structure_data),vcenter=0,vmax=np.max(structure_data)))
    for i,c in enumerate(circles):
        c.center = (phc.layers[0].shapes[i].x_cent,phc.layers[0].shapes[i].y_cent)
        c.radius = phc.layers[0].shapes[i].r
    for i,ca in enumerate(circlesAround):
        x = phc.layers[0].shapes[i].x_cent
        ca.center = (x-np.sign(x),phc.layers[0].shapes[i].y_cent)
        ca.radius = phc.layers[0].shapes[i].r

    # Update the line plot with shaded regions
    x_vals = np.linspace(kmin,kmax,25)/(2*np.pi)
    line1.set_data(x_vals, gme.freqs[:,20])
    line2.set_data(x_vals, gme.freqs[:,21])
    kpoint.set_data([x_vals[8]],[gme.freqs[8,20]])
    
    # Remove previous shaded regions if they exist
    global shade1, shade2

    # Create new shaded regions
    if frame == 0:
        shade1 = ax3.fill_between(x_vals, gme.freqs[:,19],  y2=0, color='gray', alpha=0.3)  # Shade above line1
        shade2 = ax3.fill_between(x_vals, gme.freqs[:,22],  y2=1, color='gray', alpha=0.3)  # Shade below line2

    return line, structure_plot, line1, line2, shade1, shade2, kpoint, *circles, *circlesAround

# Create animation
ani = FuncAnimation(fig, update, frames=10, init_func=init, blit=True)

# Display the animation as HTML
display(HTML(ani.to_jshtml()))
plt.close(fig)


# %%
freqs = np.array([run['freq'][0] for run in data.values()])
freqsb = np.array([run['freq'][1][0] for run in data.values()])
freqsa1 = np.array([run['freq'][2][0] for run in data.values()])
freqsa2 = np.array([run['freq'][2][1] for run in data.values()])
freqsa3 = np.array([run['freq'][2][2] for run in data.values()])

plt.plot(freqs)
plt.plot(freqsb)
plt.plot(freqsa1,'r')
plt.plot(freqsa2,'r')
plt.plot(freqsa3,'r')
# %%
from defineCrystal import W1
import legume

phcw1,_ = W1(Ny=20,Nx=1,dslab=170/266,n_slab=3.4638**2,ra=.3)
out = legume.viz.eps_xy(phcw1,Nx=20,Ny=300)
# %%
nk = 25 # Number of k-points
kmin, kmax = np.pi*.5, np.pi # Min and max k-values
path = np.vstack((np.linspace(kmin, kmax, nk), np.zeros(nk))) #k vectors

gme = legume.GuidedModeExp(phcw1,gmax=2)
gme.run(kpoints=path,numeig=50,compute_im=False)
#%%
xs = np.linspace(kmin,kmax,nk)/(2*np.pi)
plt.plot(xs,gme.freqs[:,18:22])
plt.axvline(gme.kpoints[0,8]/2/np.pi)
# plt.axhline(.295,color='r',linestyle='--')
# plt.axhline(.280,color='r',linestyle='--')
plt.xlabel('K')
plt.ylabel('Frequency')
plt.show()
# %%
ylim = 10*np.sqrt(3)/2
ys = np.linspace(-ylim,ylim,200)
fields,_,_ = gme.get_field_xy('E',20,20,phcw1.layers[0].d/2,ygrid=ys,component='xyz')
eabs = np.abs(np.conj(fields['x'])*fields['x']+np.conj(fields['y'])*fields['y']+np.conj(fields['z'])*fields['z'])
plt.imshow(eabs)
# %%
ys = np.array([s.y_cent for s in phcw1.layers[0].shapes])
print(ys[1:]-ys[:-1])
print(phcw1.lattice.a2[1]-ys[-1]+ys[0])
# %%
with open('results/alphaDng_runs/BIW_BW2.json') as file:
    out = json.load(file)
valsBIW = np.array([run['other'] for run in out['iterations'].values()])

with open('results/alphaDng_runs/W1_BW2.json') as file:
    out = json.load(file)
valsW1 = np.array([run['other'] for run in out['iterations'].values()])
valsW1 = valsW1/valsW1[0]
valsBIW = valsBIW/valsBIW[0]
fig, ax = plt.subplots()
plt.plot(valsBIW,linewidth=3,label='BIW')
plt.plot(valsW1,linewidth=3,label='W1')
plt.yscale('log')
#plt.yticks([1,.1],['1','0.1'])
plt.legend()
plt.xlabel('Iterations')
plt.ylabel(r'$(\alpha/n_g^2)/(\alpha_0/n_{g0}^2)$')
plt.show()

#%%

#------------------------------------- Final Plots ----------------------------


with open('results/alphaDng_runs/W1_BW2.json') as file:
    out = json.load(file)
metadata = out["metadata"]

#process the dx, dy, and dr values form the iteration of interest
metadata["dx"] = {eval(key):value for key,value in metadata['dx'].items()}
metadata["dy"] = {eval(key):value for key,value in metadata['dy'].items()}
metadata["dr"] = {eval(key):value for key,value in metadata['dr'].items()}

#generate crystal
phc,_ = globals()[metadata["crystal"]](**metadata)

metadata["dx"] = {eval(key):value for key,value in out['iterations']['0']['dx'].items()}
metadata["dy"] = {eval(key):value for key,value in out['iterations']['0']['dy'].items()}
metadata["dr"] = {eval(key):value for key,value in out['iterations']['0']['dr'].items()}

#original structure
phcOG,_ = globals()[metadata["crystal"]](**metadata)

#set up and run GME
nk = 100 # Number of k-points
kmin, kmax = np.pi*.5, np.pi # Min and max k-values
path = np.vstack((np.linspace(kmin, kmax, nk), np.zeros(nk))) #k vectors
gme = legume.GuidedModeExp(phc,gmax=2)
gme.run(kpoints=path,numeig=2*metadata["optMode"],compute_im=False)

#original structure
gmeOG = legume.GuidedModeExp(phcOG,gmax=2)
gmeOG.run(kpoints=path,numeig=2*metadata["optMode"],compute_im=False)

#%%
#get the two bands from inside the band gap and remove them from the array 
plt.rcParams.update({'font.size':20})
kpoints = path[0]/2/np.pi
plt.plot(kpoints,gme.freqs[:,:metadata["optMode"]],'k')
plt.plot(kpoints,gme.freqs[:,metadata["optMode"]+2:],'k')
plt.plot(kpoints,gme.freqs[:,metadata["optMode"]],color='darkviolet')
plt.plot(kpoints,gme.freqs[:,metadata["optMode"]+1],color='darkviolet',linestyle='--')
plt.fill_between(kpoints,kpoints,np.max(kpoints),color='darkGray',alpha=1)
bandmin,bandmax = .253,.2665
plt.fill_between(kpoints,bandmin,bandmax,color='cyan',alpha=.5)
plt.scatter(metadata['kpoints'][0][0]/2/np.pi,list(out['iterations'].values())[-1]["freq"][0],facecolors='none',edgecolors='darkviolet',linewidths=2,s=100)

#find the k point that is closest to the point we optomized for
kindex = (np.abs(path[0]-metadata['kpoints'][0][0])).argmin()
plt.xlim(.25,.5)
ymin, ymax = gme.freqs[kindex,metadata["optMode"]]-.05,gme.freqs[kindex,metadata["optMode"]]+.05,
plt.ylim(ymin,ymax)

#label everything
plt.xlabel(r"$k_xa/2\pi$")
plt.ylabel(r"$\omega a/2\pi c$")

#get second y axis with THz
plt.twinx()
conversion_factor = 1e-12*299792458/metadata['a']/1e-9
plt.plot(kpoints,gme.freqs[:,metadata["optMode"]]*conversion_factor,color='darkviolet')
plt.ylim(ymin*conversion_factor,ymax*conversion_factor)
plt.ylabel('Frequency [THz]')
plt.show()
#%%

#OG ------------
#get the two bands from inside the band gap and remove them from the array 
plt.rcParams.update({'font.size':20})
kpoints = path[0]/2/np.pi
plt.plot(kpoints,gmeOG.freqs[:,:metadata["optMode"]],'k')
plt.plot(kpoints,gmeOG.freqs[:,metadata["optMode"]+2:],'k')
plt.plot(kpoints,gmeOG.freqs[:,metadata["optMode"]],color='darkviolet')
plt.plot(kpoints,gmeOG.freqs[:,metadata["optMode"]+1],color='darkviolet',linestyle='--')
plt.fill_between(kpoints,kpoints,np.max(kpoints),color='darkGray',alpha=1)
bandminOG,bandmaxOG = .252,.275
plt.fill_between(kpoints,bandminOG,bandmaxOG,color='cyan',alpha=.5)
plt.scatter(metadata['kpoints'][0][0]/2/np.pi,list(out['iterations'].values())[0]["freq"][0],facecolors='none',edgecolors='darkviolet',linewidths=2,s=100)

#find the k point that is closest to the point we optomized for
kindex = (np.abs(path[0]-metadata['kpoints'][0][0])).argmin()
plt.xlim(.25,.5)
ymin, ymax = gme.freqs[kindex,metadata["optMode"]]-.05,gme.freqs[kindex,metadata["optMode"]]+.05,
plt.ylim(ymin,ymax)

#label everything
plt.xlabel(r"$k_xa/2\pi$")
plt.ylabel(r"$\omega a/2\pi c$")

#get second y axis with THz
plt.twinx()
conversion_factor = 1e-12*299792458/metadata['a']/1e-9
plt.plot(kpoints,gmeOG.freqs[:,metadata["optMode"]]*conversion_factor,color='darkviolet')
plt.ylim(ymin*conversion_factor,ymax*conversion_factor)
plt.ylabel('Frequency [THz]')
plt.show()
# %%
firstH, lastH = 13,13+16
ylim = 8*np.sqrt(3)/2
ys = np.linspace(-ylim/2,ylim/2,300)
fields,_,_ = gme.get_field_xy('E',kindex,metadata['optMode'],phc.layers[0].d/2,ygrid=ys,component='xyz')
eabs = np.abs(np.conj(fields['x'])*fields['x']+np.conj(fields['y'])*fields['y']+np.conj(fields['z'])*fields['z'])
levels = np.linspace(0,round(np.max(eabs),3),10)
cmap = plt.get_cmap("plasma",len(levels)-1)
norm = mcolors.BoundaryNorm(boundaries=levels,ncolors=cmap.N)

fig, ax = plt.subplots()
cax = ax.imshow(eabs.T,extent=[-ylim/2,ylim/2,.5,-.5],cmap=cmap,norm=norm)

colors = plt.cm.rainbow(np.linspace(0,1,lastH-firstH))
colors = np.concatenate((colors[len(colors)//2:],colors[:-len(colors)//2]))

circles = [Circle((s.y_cent,s.x_cent),s.r,edgecolor='white',facecolor='white') for s in phc.layers[0].shapes[firstH:lastH]]
circlesInner = [Circle((c.center[0],c.center[1]),c.radius*.5,facecolor=co) for c,co in zip(circles,colors)]
for c,ci in zip(circles,circlesInner):
    plt.gca().add_patch(c)
    plt.gca().add_patch(ci)
circlesAround = [Circle((0,0),edgecolor='white',facecolor='white') for _ in range(len(circles))]
for i,ca in enumerate(circlesAround):
    ca.center = (circles[i].center[0],circles[i].center[1]-np.sign(circles[i].center[1]))
    ca.radius = circles[i].radius
    plt.gca().add_patch(ca)
    plt.gca().add_patch(Circle(ca.center,ca.radius*.5,facecolor=colors[i]))

maxval = round(np.max(eabs),3)
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right",size="7%",pad=.1)
cbar = fig.colorbar(cax,cax=cbar_ax)
cbar.set_ticks([0,maxval])
cbar.set_ticklabels([0,f'{maxval}'])


ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
ax.set_title(r'$|E|^2$')

ax.set_ylim(-.5,.5)
ax.set_xlim(-ylim/2,ylim/2)

plt.show()

#%%
#--------------------------------------OG-----------------------------
fields,_,_ = gmeOG.get_field_xy('E',kindex,metadata['optMode'],phc.layers[0].d/2,ygrid=ys,component='xyz')
eabs = np.abs(np.conj(fields['x'])*fields['x']+np.conj(fields['y'])*fields['y']+np.conj(fields['z'])*fields['z'])
levels = np.linspace(0,round(np.max(eabs),3),10)
cmap = plt.get_cmap("plasma",len(levels)-1)
norm = mcolors.BoundaryNorm(boundaries=levels,ncolors=cmap.N)

fig, ax = plt.subplots()
cax = ax.imshow(eabs.T,extent=[-ylim/2,ylim/2,.5,-.5],cmap=cmap,norm=norm)

colors = plt.cm.rainbow(np.linspace(0,1,lastH-firstH))
colors = np.concatenate((colors[len(colors)//2:],colors[:-len(colors)//2]))

circles = [Circle((s.y_cent,s.x_cent),s.r,edgecolor='white',facecolor='white') for s in phcOG.layers[0].shapes[firstH:lastH]]
circlesInner = [Circle((c.center[0],c.center[1]),c.radius*.5,facecolor=co) for c,co in zip(circles,colors)]
for c,ci in zip(circles,circlesInner):
    plt.gca().add_patch(c)
    plt.gca().add_patch(ci)
circlesAround = [Circle((0,0),edgecolor='white',facecolor='white') for _ in range(len(circles))]
for i,ca in enumerate(circlesAround):
    ca.center = (circles[i].center[0],circles[i].center[1]-np.sign(circles[i].center[1]))
    ca.radius = circles[i].radius
    plt.gca().add_patch(ca)
    plt.gca().add_patch(Circle(ca.center,ca.radius*.5,facecolor=colors[i]))

maxval = round(np.max(eabs),3)
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right",size="7%",pad=.1)
cbar = fig.colorbar(cax,cax=cbar_ax)
cbar.set_ticks([0,maxval])
cbar.set_ticklabels([0,f'{maxval}'])


ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
ax.set_title(r'$|E|^2$')

ax.set_ylim(-.5,.5)
ax.set_xlim(-ylim/2,ylim/2)

plt.show()

# %%
from inverseDesign import compute_alphaDng
ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gme.kpoints[:,1:]-gme.kpoints[:,:-1],axis=0)/(gme.freqs[1:,metadata['optMode']]-gme.freqs[:-1,metadata['optMode']]))

alpha = np.zeros(nk)
for i in range(nk):
    alpha[i] = compute_alphaDng(gme,phc,i,metadata["optMode"],metadata["a"],phc.layers[0].d/2)

plt.plot(gme.freqs[:-1,metadata['optMode']]*conversion_factor,ng,color='darkviolet')
plt.axvspan(bandmin*conversion_factor,bandmax*conversion_factor,color='cyan')
plt.axvspan(280,bandmin*conversion_factor,color='darkgray')
plt.scatter(list(out['iterations'].values())[-1]["freq"][0]*conversion_factor,ng[kindex],facecolors='none',edgecolors='darkviolet',linewidths=2,s=100)
plt.ylabel(r'$n_g$',color='darkviolet')
plt.tick_params(axis='y',colors='darkviolet')
plt.ylim(0,100)
plt.xlabel("Frequency [THz]")

plt.twinx()
plt.plot(gme.freqs[:-1,metadata['optMode']]*conversion_factor,(alpha[1:]*ng**2)/metadata['a']/1e-9*1e-3,color='Blue')
plt.ylabel('alpha',color='blue')
plt.tick_params(axis='y',colors='blue')
plt.yscale('log')
plt.ylim(.001,100)


plt.xlim(282.5,302.5)
plt.show()
print(ng[kindex])
#%%
ngOG = (1/(2*np.pi))*np.abs(np.linalg.norm(gmeOG.kpoints[:,1:]-gmeOG.kpoints[:,:-1],axis=0)/(gmeOG.freqs[1:,metadata['optMode']]-gmeOG.freqs[:-1,metadata['optMode']]))

alphaOG = np.zeros(nk)
for i in range(nk):
    alphaOG[i] = compute_alphaDng(gmeOG,phcOG,i,metadata["optMode"],metadata["a"],phc.layers[0].d/2)

#%%
plt.plot(gmeOG.freqs[:-1,metadata['optMode']]*conversion_factor,ngOG,color='darkviolet')
plt.axvspan(bandminOG*conversion_factor,bandmaxOG*conversion_factor,color='cyan')
plt.axvspan(277.5,bandminOG*conversion_factor,color='darkgray')
plt.scatter(list(out['iterations'].values())[0]["freq"][0]*conversion_factor,ngOG[kindex],facecolors='none',edgecolors='darkviolet',linewidths=2,s=100)
plt.ylabel(r'$n_g$',color='darkviolet')
plt.tick_params(axis='y',colors='darkviolet')
plt.ylim(0,100)
plt.xlabel("Frequency [THz]")

plt.twinx()
plt.plot(gmeOG.freqs[:-1,metadata['optMode']]*conversion_factor,(alphaOG[1:]*ngOG**2)/metadata['a']/1e-9*1e-3,color='Blue')
plt.ylabel(r'$\langle\alpha\rangle$',color='blue')
plt.tick_params(axis='y',colors='blue')
plt.yscale('log')
plt.ylim(.001,100)

plt.xlim(277.5,317.5)
plt.show()
print(ngOG[kindex])
# %%
from inverseDesign import holeBorders,get_xyfield

holes,_,_ = holeBorders(phc,phidiv=100)
fields = get_xyfield(gme,kindex,metadata["optMode"],holes)
absE = np.abs(np.conj(fields['x'])*fields['x']+np.conj(fields['y'])*fields['y']+np.conj(fields['z'])*fields['z'])

holesOG,_,_ = holeBorders(phcOG,phidiv=100)
fieldsOG = get_xyfield(gmeOG,kindex,metadata["optMode"],holesOG)
absEOG = np.abs(np.conj(fieldsOG['x'])*fieldsOG['x']+np.conj(fieldsOG['y'])*fieldsOG['y']+np.conj(fieldsOG['z'])*fieldsOG['z'])

# %%
xs = np.linspace(-np.pi,np.pi,len(absE[0]))
for i,C in enumerate(absE[firstH:lastH]):
    plt.plot(xs,C,color=colors[i])
    plt.plot(xs,absEOG[i+firstH],color=colors[i],linestyle='--')
plt.xticks([-np.pi,np.pi],[r'$-\pi$',r'$\pi$'])
plt.xlabel(r'$\theta$')
plt.ylabel(r'$|E|^2$ $[a^{-3}]$')
plt.show()
# %%
with open('results/alphaDng_runs/BIW_BW2.json') as file:
    outBIW = json.load(file)


with open('results/alphaDng_runs/W1_BW2.json') as file:
    outW1 = json.load(file)

BIWLoss = 10*np.log10(1/(1-100*np.array([run['other'] for run in outBIW['iterations'].values()])))/outBIW['metadata']['a']/1e-9*1e-3
W1Loss = 10*np.log10(1/(1-36*np.array([run['other'] for run in outW1['iterations'].values()])))/outW1['metadata']['a']/1e-9*1e-3

plt.plot(BIWLoss,label='BIW')
plt.plot(W1Loss,label='W1')
plt.yscale('log')
plt.xlabel('Iteratoin')
plt.ylabel('Loss [dB/cm]')
plt.legend()
plt.show()
# %%
def examp(**kwargs):
    print(kwargs["hello"])

examp(hello="hello")
# %%
plt.plot(ngOG,1/(alphaOG[1:]/phcOG.layers[0].d))
plt.xlim(1,300)
plt.xscale('log')
plt.yscale('log')
# %%
1e-1
# %%
np.arange(3)
for i in [0,1,2]:
    f = lambda 