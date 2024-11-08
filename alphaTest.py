#%%
from defineCrystal import TopoWave, W1
import legume
import numpy as np
import matplotlib.pyplot as plt
#%%
#toni's crystal
a = 266 # nm, the lattice constant (pitch) of the crystal
h = 170 # nm, the height (depth) of the slab
r1, r2 = .235*a, .105*a # nm, starting radii

nk = 500 # Number of k-points
kmin, kmax = np.pi*.5, np.pi # Min and max k-values
path = np.vstack((np.linspace(kmin, kmax, nk), np.zeros(nk))) #k vectors

phc,_ = TopoWave(Ny=21,n_slab=3.4638**2,ra=(r2/a,r1/a),dslab=h/a)
#legume.viz.eps_xy(phc,Nx=200,Ny=200)
#%%
gme = legume.GuidedModeExp(phc,gmax=2)
gme.run(kpoints=path,numeig=23,compute_im=False)
#%%
xs = np.linspace(kmin,kmax,nk)/(2*np.pi)
plt.plot(xs,gme.freqs)
plt.show()
# %%
#group index
n = 41
ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gme.kpoints[:,1:]-gme.kpoints[:,:-1],axis=0)/(gme.freqs[1:,n]-gme.freqs[:-1,n]))
plt.plot(gme.freqs[1:,n],ng)
n = 40
ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gme.kpoints[:,1:]-gme.kpoints[:,:-1],axis=0)/(gme.freqs[1:,n]-gme.freqs[:-1,n]))
plt.plot(gme.freqs[1:,n],ng)
plt.yscale('log')
plt.ylim(1,10**2)
plt.show()

# %%

#get the hole borders
def holeBorders(phc,phidiv=20):

    #set up some variables
    shapes = phc.layers[0].shapes
    phis = np.linspace(0,2*np.pi,phidiv,endpoint=False)
    cphis = np.cos(phis);sphis = np.sin(phis)

    #get the list of hole atributes
    holeCords = np.zeros((len(shapes),3))
    for i,s in enumerate(shapes):
        holeCords[i] = np.array([s.x_cent,s.y_cent,s.r])
    
    #get the coordiates of the hole borders
    borders = np.zeros((len(shapes),phidiv,2))
    for i,h in enumerate(holeCords):
        borders[i] = np.array([h[0]+h[2]*cphis,h[1]+h[2]*sphis]).T

        #adjust the values so that they loop in the BZ
        borders[i,:,0][borders[i,:,0]>phc.lattice.a1[0]/2] -= phc.lattice.a1[0]
        borders[i,:,1][borders[i,:,1]>phc.lattice.a2[1]/2] -= phc.lattice.a2[1]
    
    return(borders,phis,holeCords[:,2])

#get the field for arbitrary xy points 
def get_xyfield(gme,k,n,xys,field='E',z=0,component='xyz'):

    #setup
    ft = {}
    ft['x'],ft['y'],ft['z'] = gme.ft_field_xy(field,k,n,z)
    fis = {}
    _,ind_unique = np.unique(gme.gvec,return_index=True,axis=1)

    #loop through adding the field
    for comp in component:
        if comp in ft.keys():
            if not (comp in fis.keys()):
                fis[comp] = np.zeros(xys[:,:,0].shape,dtype=np.complex128)
                for indg in ind_unique:
                    fis[comp] += np.sqrt(np.pi)*ft[comp][indg]*np.exp(1j*gme.gvec[0,indg]*xys[:,:,0]+1j*gme.gvec[1,indg]*xys[:,:,1])
        else:
            raise ValueError("component can be any combination of xyz")
    
    return(fis)
#%%
#compute the polorization dot field terms 
def compute_pdote(gme,phc,k,n,z,hbs,phis):
    
    #get the field components
    E = get_xyfield(gme,k,n,hbs,z=z)
    D = get_xyfield(gme,k,n,hbs,field='D',z=z,component='xy')

    #get the parrallel E components and the perpendicular D components
    #currently this ignores the z components (need to ask toni)
    Epara = np.array([-np.sin(phis)*E['x'],np.cos(phis)*E['y'],E['z']])/np.sqrt(2)
    Dperp = np.array([np.cos(phis)*D['x'],np.sin(phis)*D['y'],np.zeros_like(E['z'])])

    #get the neccicary coefficents 
    eps_b = phc.layers[0].eps_b
    eps_h = 1

    #polorizability 
    p = Epara+(eps_b+eps_h)*Dperp/(2*eps_b*eps_h)

    #now take th polorizability dotted with E
    pde_r = np.conj(E['x'])*np.conj(p[0])+np.conj(E['y'])*np.conj(p[1])+np.conj(E['z'])*np.conj(p[2])
    pde_rp = (E['x']*p[0]+E['y']*p[1]+E['z']*p[2])
    
    return(pde_r,pde_rp)


#comput alpha
def compute_alpha(gme,phc,k,n,z,phidiv=15,lp=40,sig=3,zsplit=100):

    #get the hole bourders
    hbs, phis, hole_rs = holeBorders(phc,phidiv=phidiv)

    #get the pdotE terms 
    for i in range(zsplit):
        pde_r,pde_rp = compute_pdote(gme,phc,k,n,z*2*(i+.5)/zsplit,hbs,phis)

        #Do the multiplicatoin for the p dot e part. We will add jacobian determinate after
        pde_meshs = np.array([np.meshgrid(pde_r[i],pde_rp[i]) for i in range(pde_r.shape[0])])
        preSum_pde = pde_meshs[:,0]*pde_meshs[:,1]

        #the real exponential term
        phi_mesh,phi_p_mesh = np.meshgrid(phis,phis)
        real_exp = (np.abs(phi_mesh-phi_p_mesh)*(-hole_rs[:,np.newaxis,np.newaxis]))/(lp) #nm terms cancle on holes and lp

        #the imaginary exponential term
        x_meshs = np.array([np.meshgrid(hbs[i,:,0],hbs[i,:,0]) for i in range(hbs.shape[0])])
        imag_exp = 2*(np.linalg.norm(gme.kpoints[:,k]))*(x_meshs[:,0]-x_meshs[:,1]) #a terms cancle

        #we want to run intigral, inculding the jacobian determinant
        intigrand = preSum_pde*np.exp(real_exp+1j*imag_exp)
        intigral = np.sum(intigrand,axis=(1,2))*(hole_rs*np.pi*2/phidiv)**2
        
        #calculate leading coeficents for each circle in sum
        #coeficents become 
        #ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gme.kpoints[:,k]-gme.kpoints[:,k-1])/(gme.freqs[k,n]-gme.freqs[k-1,n]))
        circle_coeffs = ((.3*2*np.pi)*a*gme.freqs[k,n]*sig*(phc.layers[0].eps_b-1)/2)**2

        #compute sum and return
        out += intigral.shape[0]*circle_coeffs*np.sum(intigral)*(a*10**-9)**2

    return(np.real(out*z*2/zsplit))
#%%
k = 63
n=20
field='E'
z=phc.layers[0].d/2

vals = np.zeros(gme.kpoints.shape[1])
vals2 = np.zeros(gme.kpoints.shape[1])

for k in range(gme.kpoints.shape[1]):
    if k%25 ==0:
        print(k)
    vals[k] = compute_alpha(gme,phc,k,n,z)
    vals2[k] = compute_alpha(gme,phc,k,n+1,z)
#%%
plt.rcParams.update({'font.size':14})
plt.plot(gme.freqs[1:,40],1/np.real(vals[1:]))
plt.plot(gme.freqs[1:,41],1/np.real(vals2[1:]))
plt.ylabel(r'$L_{back}$')
plt.xlabel(r'$\omega\ [2\pi c/a]$')
plt.ylim(1,10**6)
plt.yscale('log')
plt.show()
#%%


phcw1,_ = W1(Ny=40,Nx=1,dslab=170/266,n_slab=3.4638**2,ra=.3)
out = legume.viz.eps_xy(phcw1,Nx=20,Ny=300)
# %%
nk = 200 # Number of k-points
kmin, kmax = np.pi*.5, np.pi # Min and max k-values
path = np.vstack((np.linspace(kmin, kmax, nk), np.zeros(nk))) #k vectors

gmeW1 = legume.GuidedModeExp(phcw1,gmax=5)
gmeW1.run(kpoints=path,numeig=50,compute_im=False)
#%%
xs = np.linspace(kmin,kmax,nk)/(2*np.pi)
plt.plot(xs,gmeW1.freqs)
plt.axvline(gmeW1.kpoints[0,8]/2/np.pi)
# plt.axhline(.295,color='r',linestyle='--')
# plt.axhline(.280,color='r',linestyle='--')
plt.xlabel('K')
plt.ylabel('Frequency')
plt.show()
#%%
n = 21
ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gmeW1.kpoints[:,1:]-gmeW1.kpoints[:,:-1],axis=0)/(gmeW1.freqs[1:,n]-gmeW1.freqs[:-1,n]))
plt.plot(gmeW1.freqs[1:,n],ng)
n = 20
ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gmeW1.kpoints[:,1:]-gmeW1.kpoints[:,:-1],axis=0)/(gmeW1.freqs[1:,n]-gmeW1.freqs[:-1,n]))
plt.plot(gmeW1.freqs[1:,n],ng)
plt.yscale('log')
plt.ylim(1,10**2)
plt.show()
# %%
n=40
field='E'
z=phcw1.layers[0].d/2

#valsW1 = np.zeros(gmeW1.kpoints.shape[1])
#vals2W1 = np.zeros(gmeW1.kpoints.shape[1])
#valsGME4W1 = np.zeros(gmeW1.kpoints.shape[1])
valsGME5W1 = np.zeros(gmeW1.kpoints.shape[1])

for k in range(gmeW1.kpoints.shape[1]):
    if k%25 ==0:
        print(k)
    valsGME5W1[k] = compute_alpha(gmeW1,phcw1,k,n,z,phidiv = 100)
    #valsGME4W1[k] = compute_alpha(gmeW1,phcw1,k,n,z,phidiv = 15)
    #valsW1[k] = compute_alpha(gmeW1,phcw1,k,n,z,phidiv = 100)
    #vals2W1[k] = compute_alpha(gmeW1,phcw1,k,n,z,phidiv = 15)
    #vals2W1[k] = compute_alpha(gmeW1,phcw1,k,n+1,z,phidiv = 30)
# %%
plt.rcParams.update({'font.size':14})
n = 40
ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gmeW1.kpoints[:,1:]-gmeW1.kpoints[:,:-1],axis=0)/(gmeW1.freqs[1:,n]-gmeW1.freqs[:-1,n]))
plt.plot(ng,1/np.real(valsGME4W1[1:]))
plt.plot(ng,1/np.real(valsGME5W1[1:]),'--')
plt.yscale('log')
plt.xscale('log')
plt.show()
#%%
print(valsGME5W1[1:])
# %%
plt.plot(np.real(vals),'o')
plt.plot(np.real(vals2),'o')
#%%
s, e = 123,130
plt.plot(vals[s-2:e],'o')
plt.plot(vals2[s-2:e],'o')
# %%
freq1 = np.concatenate((gme.freqs[:s+1,20],gme.freqs[s+1:,21]))
freq2 = np.concatenate((gme.freqs[:s+1,21],gme.freqs[s+1:,20]))
plt.plot(freq1[s-2:e],'o')
plt.plot(freq2[s-2:e],'o')

# %%
alpha1 = np.concatenate((vals[:s+1],vals2[s+1:]))
alpha2 = np.concatenate((vals2[:s+1],vals[s+1:]))
plt.plot(alpha1[s-2:e],'o')
plt.plot(alpha2[s-2:e],'o')
# %%
plt.rcParams.update({'font.size':16})
n = 41
ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gme.kpoints[:,1:]-gme.kpoints[:,:-1],axis=0)/(freq1[1:]-freq1[:-1]))
plt.plot(ng,1/np.real(alpha1[1:]),color='limegreen',linewidth=3,label='BIW 1')
n = 40
ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gme.kpoints[:,1:]-gme.kpoints[:,:-1],axis=0)/(freq2[1:]-freq2[:-1]))
plt.plot(ng,1/np.real(alpha2[1:]),color='green',linewidth=3,label='BIW 2')
n = 40
ng = (1/(2*np.pi))*np.abs(np.linalg.norm(gmeW1.kpoints[:,1:]-gmeW1.kpoints[:,:-1],axis=0)/(gmeW1.freqs[1:,n]-gmeW1.freqs[:-1,n]))
plt.plot(ng,1/np.real(valsW1[1:]),color='orange',linewidth=3,label='W1')
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$L_{back}n_g^2$')
plt.xlabel(r'$n_g$')
plt.xlim(1,400)
plt.ylim(1E5,1E7)
plt.legend()
plt.show()
# %%

ngalt = np.zeros_like(gme.freqs[:,0])
ngalt2 = np.zeros_like(gme.freqs[:,0])
ngalt3 = np.zeros_like(gme.freqs[:,0])
for i in range(ngalt.size):

    Nx,Ny = 100,100

    Efield,_,_ = gme.get_field_xy('E',i,20,phc.layers[0].d/2,Nx=Nx,Ny=Ny)
    Hfield,_,_ = gme.get_field_xy('H',i,20,phc.layers[0].d/2,Nx=Nx,Ny=Ny)

    Efield = np.array([[Efield['x']],[Efield['y']],[Efield['z']]])
    Hfield = np.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])
    
    ngalt[i]= -np.sum(np.real(np.cross(np.conj(Efield),Hfield,axis=0)))*phc.lattice.a2[1]/Nx/Ny*phc.layers[0].d
    
    Nx,Ny = 5,40

    Efield,_,_ = gme.get_field_xy('E',i,20,phc.layers[0].d/2,Nx=Nx,Ny=Ny)
    Hfield,_,_ = gme.get_field_xy('H',i,20,phc.layers[0].d/2,Nx=Nx,Ny=Ny)

    Efield = np.array([[Efield['x']],[Efield['y']],[Efield['z']]])
    Hfield = np.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])

    ngalt2[i]= -np.sum(np.real(np.cross(np.conj(Efield),Hfield,axis=0)))*phc.lattice.a2[1]/Nx/Ny*phc.layers[0].d

    Nx,Ny = 5,30

    Efield,_,_ = gme.get_field_xy('E',i,20,phc.layers[0].d/2,Nx=Nx,Ny=Ny)
    Hfield,_,_ = gme.get_field_xy('H',i,20,phc.layers[0].d/2,Nx=Nx,Ny=Ny)
    Efield = np.array([[Efield['x']],[Efield['y']],[Efield['z']]])
    Hfield = np.array([[Hfield['x']],[Hfield['y']],[Hfield['z']]])

    ngalt3[i]= -np.sum(np.real(np.cross(np.conj(Efield),Hfield,axis=0)))*phc.lattice.a2[1]/Nx/Ny*phc.layers[0].d
# %%
n = 20
ng = (1/(2*np.pi))*np.linalg.norm(gme.kpoints[:,:-1]-gme.kpoints[:,1:],axis=0)/(gme.freqs[:-1,n]-gme.freqs[1:,n])

plt.plot(gme.freqs[:,n],ngalt)
plt.plot(gme.freqs[:,n],ngalt2)
plt.plot(gme.freqs[:,n],ngalt3)
plt.plot(gme.freqs[1:,n],-1/ng)
plt.show()
# %%
out = np.zeros(100,dtype="float")
xs = np.linspace(1,10,100)
for i,v in enumerate(xs):
    out[i] = np.sum((ngalt[:-1]/v-1/ng)**2)
plt.plot(xs,out)
# %%
plt.plot