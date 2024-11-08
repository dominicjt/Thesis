#file for generating constraints and bounds 
import numpy as np
from inverseDesign import placeParams
import autograd.numpy as npa
from inverseDesign import costWrapper
from inverseDesign import of_Q


#function for generating L3 constraints. all inequality constraints are stated as >=0
def L3const(minfreq=0,maxfreq=1000,minrad=0,mindist=0,dx={},dy={},dr={},ra=.25,objective_function=of_Q,**kwargs):
    
    #set constraints initally empty
    constraints = []

    #get the combined length of all dictionaries, this is useful multiple times
    totlength = len(dx)+len(dy)+len(dr)

    #generate the indexes that should be used for the constraint functions
    indexs = np.arange(totlength)
    dx,dy,dr = placeParams(indexs,dx=dx,dy=dy,dr=dr)

    #generate constraints for the radius 
    for key, i in dr.items():
        constFunc = lambda x,i=i: x[i]-minrad+ra
        constraints.append({'type':'ineq','fun':constFunc})

    #generate constraints that hold the holes in their unit cell
    for key, i in dx.items():
        constFunc = lambda x,i=i: 1-npa.abs(x[i])
        constraints.append({'type':'ineq','fun':constFunc})
    for key, i in dy.items():
        constFunc = lambda x,i=i: 1-npa.abs(x[i])
        constraints.append({'type':'ineq','fun':constFunc})


    
    #we now generate constraints for holes that are close to eachother. we want one 
    #constraint per pair, and it should have constraints both between two moving holes 
    #and one moving hole and one statinoary hole
    #start by getting a list of all holes that that have parameters being altered
    movingHoles = []
    seenHoles = set()
    for dictionary in [dx,dy,dr]:
        for key, _ in dictionary.items():
            if key not in seenHoles:
                seenHoles.add(key)
                movingHoles.append(key)
    
    #now sort moving holes so that it moves from the top right down,
    #this is done by sorting by the first element of the keys tupple then the second
    movingHoles.sort(key=lambda x: (x[0], x[1]), reverse=True)

    #now we loop over the elements in key and generate their constraints.
    #we generate a constraint for any hole that is of a lessor sorting then the hole
    #since they haven't had any constraints generated for them. Then, for any hole with a larger
    #index, we check if the hole is in the moving holes, if it is then the constraint is already 
    #generated so we skip. obviously don't generate constraints for non-moving holes or 
    #holes that are not in the close neighborhood. the neighborhood is depenedent on how the 
    #indexes are set up.
    for hole in movingHoles:

        #get the index values of the different parameters associated with the holes
        xIndex = dx.get(hole, totlength)
        yIndex = dy.get(hole, totlength)
        rIndex = dr.get(hole, totlength)

        #get the original location for the moving hole we are looking at 
        #get position
        if hole[1]%2==1:
            mx = hole[0] + 0.5
        else:
            mx = hole[0]
        my = hole[1]*np.sqrt(3)/2

        #get the list of larger and lesser holes according to the ordering defined above
        largerNs = [(hole[0]+1,hole[1]),(hole[0],hole[1]+1)]
        smallerNs = [(hole[0],hole[1]-1),(hole[0]-1,hole[1]+1),(hole[0]-1,hole[1]),(hole[0]-1,hole[1]-1)]
        
        #generate the constraint functions by getting the position of the hole by adding the 
        #dy,dy,dr values then getting the starting location
        for neighbor in largerNs:

            #skip if not a real hole
            if neighbor[1]==0 and (neighbor[0]==-1 or neighbor[0]==0 or neighbor[0]==1):
                continue

            #skip if larger and in moving list since inequality is already made
            if neighbor in movingHoles:
                continue

            #get position
            if neighbor[1]%2==1:
                nx = neighbor[0] + 0.5
            else:
                nx = neighbor[0]
            ny = neighbor[1]*np.sqrt(3)/2
            
            #make constraint function for this value. make it so that if the parameter is not being changed then 
            #the appended zero is picked.
            constFunc = lambda x,mx=mx,my=my,xIndex=xIndex,yIndex=yIndex,nx=nx,ny=ny: npa.sqrt((mx+np.append(x,0)[xIndex]-nx)**2+(my-np.append(x,0)[yIndex]-ny)**2)-(ra+np.append(x,0)[rIndex])-ra-mindist
            constraints.append({'type':'ineq','fun':constFunc})

        #do the same thing for the smaller holes. this time we have to find the dx,dy,dr values
        #for the neighboring holes if they are in the moving list
        for neighbor in smallerNs:

            #skip if not a real hole
            if neighbor[1]==0 and (neighbor[0]==-1 or neighbor[0]==0 or neighbor[0]==1):
                continue

            #get the index for the changing properties of the neighboring holes 
            if neighbor in movingHoles:
                nxIndex = dx.get(neighbor, totlength)
                nyIndex = dy.get(neighbor, totlength)
                nrIndex = dr.get(neighbor, totlength)
            else:
                nxIndex = nyIndex = nrIndex = totlength

            #get position
            if neighbor[1]%2==1:
                nx = neighbor[0] + 0.5
            else:
                nx = neighbor[0]
            ny = neighbor[1]*np.sqrt(3)/2

            #make constraint function for this value. make it so that if the parameter is not being changed then 
            #the appended zero is picked. This is done by setting the various indexs to totlength.
            constFunc = lambda x,mx=mx,my=my,xIndex=xIndex,yIndex=yIndex,nx=nx,ny=ny,nxIndex=nxIndex,nyIndex=nyIndex: npa.sqrt((mx+np.append(x,0)[xIndex]-(nx+np.append(x,0)[nxIndex]))**2+(my-np.append(x,0)[yIndex]-(ny+np.append(x,0)[nyIndex]))**2)-(ra+np.append(x,0)[rIndex])-(ra+np.append(x,0)[nrIndex])-mindist
            constraints.append({'type':'ineq','fun':constFunc})

    #generate constraints for min and max frequency        
    #greater then the minimum frequency
    constFunc = lambda x: costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1]-minfreq
    constraints.append({'type':'ineq','fun':constFunc})

    #less then the maximum frequency
    constFunc = lambda x: maxfreq-costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1]
    constraints.append({'type':'ineq','fun':constFunc})
    
    #return constraints 
    return(constraints)

def TopoWavconst(minfreq=0,maxfreq=1000,minrad=0,mindist=0,dx={},dy={},dr={},ra=(.235,.105),objective_function=of_Q,**kwargs):
    
    #set constraints initally empty
    constraints = []

    #get the combined length of all dictionaries, this is useful multiple times
    totlength = len(dx)+len(dy)+len(dr)

    #unpack ra
    r0,r1 = ra[0],ra[1]

    #generate the indexes that should be used for the constraint functions
    indexs = np.arange(totlength)
    dx,dy,dr = placeParams(indexs,dx=dx,dy=dy,dr=dr)

    #generate constraints for the radius 
    for key, i in dr.items():
        constFunc = lambda x,i=i: x[i]-minrad+npa.min([r1,r0])
        constraints.append({'type':'ineq','fun':constFunc})

    #generate constraints that hold the holes in their unit cell
    #this prevents the results from blowing up
    for key, i in dx.items():
        constFunc = lambda x,i=i: .5-npa.abs(x[i])
        constraints.append({'type':'ineq','fun':constFunc})
    for key, i in dy.items():
        constFunc = lambda x,i=i: .5-npa.abs(x[i])
        constraints.append({'type':'ineq','fun':constFunc})

    #we now generate constraints for holes that are close to eachother. we want one 
    #constraint per pair, and it should have constraints both between two moving holes 
    #and one moving hole and one statinoary hole
    #start by getting a list of all holes that that have parameters being altered
    movingHoles = []
    seenHoles = set()
    for dictionary in [dx,dy,dr]:
        for key, _ in dictionary.items():
            if key not in seenHoles:
                seenHoles.add(key)
                movingHoles.append(key)
    
    #sort moving holes so that they go form the bottom up
    movingHoles.sort(key=lambda x: (x[1], -x[2]))

    #We want to do nearest neighbor constarints. we only have to prevent the hole below
    #from interacting with the hole above it, but we need to do it such that it won't
    #loop around either
    for hole in movingHoles:

        #get the indexes for the variables we are changing in the hole we are focused on
        dx1 = dx.get(hole, totlength)
        dy1 = dy.get(hole, totlength)
        dr1 = dr.get(hole, totlength)

        #we have two cases: 0, when its the top hole, and 1 when its the bottom hole
        if hole[2]==0:

            #get the indexes for the variables we are changing in the hole above
            holeAbove = (hole[0],hole[1]+1,1)
            dx2 = dx.get(holeAbove, totlength)
            dy2 = dy.get(holeAbove, totlength)
            dr2 = dr.get(holeAbove, totlength)
        
            #the center hole is a double r1 hole, the rest will have one r1 and one r2
            if hole==(0,0,0):
                rtot = 2*r1
            else:
                rtot = r1+r0

            #generate constraints, need normal and one that prevents loops
            constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2,rtot=rtot: 
                npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2]-.5)**2+
                         (np.append(x,0)[dy1]-np.append(x,0)[dy2]-np.sqrt(3)/6)**2)-
                rtot-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
            constraints.append({'type':'ineq','fun':constFunc})

            constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2,rtot=rtot: 
                npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2]+.5)**2+
                         (np.append(x,0)[dy1]-np.append(x,0)[dy2]-np.sqrt(3)/6)**2)-
                rtot-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
            constraints.append({'type':'ineq','fun':constFunc})

        #second case, where hole is directly above
        elif hole[2]==1:

            #get the indexes for the variables we are changing in the hole above
            holeAbove = (hole[0],hole[1],0)
            dx2 = dx.get(holeAbove, totlength)
            dy2 = dy.get(holeAbove, totlength)
            dr2 = dr.get(holeAbove, totlength)

            #generate constraint func, we only need one case
            constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2: 
                npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2])**2+
                         (np.append(x,0)[dy1]-np.append(x,0)[dy2]-np.sqrt(3)/3)**2)-
                r0-r1-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
            constraints.append({'type':'ineq','fun':constFunc})
    

    #We need to set up the constraints for the bottom hole and the one below it
    #this is a lot of extra code, but it does the same thing as above
    hole = movingHoles[0]
    dx1 = dx.get(hole, totlength)
    dy1 = dy.get(hole, totlength)
    dr1 = dr.get(hole, totlength)
    if hole[2] == 0:

        #get the indexes for the variables we are changing in the hole below
        holeBelow = (hole[0],hole[1],1)
        dx2 = dx.get(holeBelow, totlength)
        dy2 = dy.get(holeBelow, totlength)
        dr2 = dr.get(holeBelow, totlength)

        #generate constraint func, we only need one case
        constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2: 
            npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2])**2+
                     (np.append(x,0)[dy1]-np.append(x,0)[dy2]-np.sqrt(3)/3)**2)-
            r0-r1-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
        constraints.append({'type':'ineq','fun':constFunc})
    
    else:
        #get the indexes for the variables we are changing in the hole below
        holeBelow = (hole[0],hole[1]-1,0)
        dx2 = dx.get(holeBelow, totlength)
        dy2 = dy.get(holeBelow, totlength)
        dr2 = dr.get(holeBelow, totlength)

        #generate constraints, need normal and one that prevents loops
        constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2: 
            npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2]-.5)**2+
                     (np.append(x,0)[dy1]-np.append(x,0)[dy2]-np.sqrt(3)/6)**2)-
            r0-r1-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
        constraints.append({'type':'ineq','fun':constFunc})

        constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2: 
            npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2]+.5)**2+
                     (np.append(x,0)[dy1]-np.append(x,0)[dy2]-np.sqrt(3)/6)**2)-
            r0-r1-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
        constraints.append({'type':'ineq','fun':constFunc})

    #---------All spacial constraints are done--------

    #generate constraints for min and max frequency        
    #greater then the minimum frequency
    constFunc = lambda x: costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1]-minfreq
    constraints.append({'type':'ineq','fun':constFunc})

    #less then the maximum frequency
    constFunc = lambda x: maxfreq-costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1]
    constraints.append({'type':'ineq','fun':constFunc})

    return(constraints)



#constraints for W1
def W1const(minfreq=0,maxfreq=1000,minrad=0,mindist=0,dx={},dy={},dr={},ra=.3,objective_function=of_Q,bandwidth=0,**kwargs):
    
    #set constraints initally empty
    constraints = []

    #get the combined length of all dictionaries, this is useful multiple times
    totlength = len(dx)+len(dy)+len(dr)

    #generate the indexes that should be used for the constraint functions
    indexs = np.arange(totlength)
    dx,dy,dr = placeParams(indexs,dx=dx,dy=dy,dr=dr)

    #generate constraints for the radius 
    for key, i in dr.items():
        constFunc = lambda x,i=i: (x[i]+ra-(.5-mindist/2-minrad)/2-minrad)+(.5-mindist/2-maxfreq-minrad)/2
        constraints.append({'type':'ineq','fun':constFunc})
        # constFunc = lambda x,i=i: .5-mindist/2-x[i]-ra
        # constraints.append({'type':'ineq','fun':constFunc})

    #generate constraints that hold the holes in their unit cell
    #this prevents the results from blowing up
    for key, i in dx.items():
        constFunc = lambda x,i=i: .5-npa.abs(x[i])
        constraints.append({'type':'ineq','fun':constFunc})
    for key, i in dy.items():
        constFunc = lambda x,i=i: .5-npa.abs(x[i])
        constraints.append({'type':'ineq','fun':constFunc})

    #we now generate constraints for holes that are close to eachother. we want one 
    #constraint per pair, and it should have constraints both between two moving holes 
    #and one moving hole and one statinoary hole
    #start by getting a list of all holes that that have parameters being altered
    movingHoles = []
    seenHoles = set()
    for dictionary in [dx,dy,dr]:
        for key, _ in dictionary.items():
            if key not in seenHoles:
                seenHoles.add(key)
                movingHoles.append(key)
    
    #sort moving holes so that they go form the bottom up
    movingHoles.sort(key=lambda x: x[1])

    #We want to do nearest neighbor constarints. we only have to prevent the hole below
    #from interacting with the hole above it, but we need to do it such that it won't
    #loop around either
    for hole in movingHoles:

        #get the indexes for the variables we are changing in the hole we are focused on
        dx1 = dx.get(hole, totlength)
        dy1 = dy.get(hole, totlength)
        dr1 = dr.get(hole, totlength)

        #get the indexes for the variables we are changing in the hole above
        holeAbove = (hole[0],hole[1]+1)
        if hole[1] == -1:
            holeAbove = (hole[0],1)
        dx2 = dx.get(holeAbove, totlength)
        dy2 = dy.get(holeAbove, totlength)
        dr2 = dr.get(holeAbove, totlength)

        #generate constraints, need normal and one that prevents loops
        constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2: 
            npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2]-.5)**2+
                     (np.append(x,0)[dy1]-np.append(x,0)[dy2]-np.sqrt(3)/2)**2)-
            2*ra-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
        constraints.append({'type':'ineq','fun':constFunc})

        constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2: 
            npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2]+.5)**2+
                     (np.append(x,0)[dy1]-np.append(x,0)[dy2]-np.sqrt(3)/2)**2)-
            2*ra-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
        constraints.append({'type':'ineq','fun':constFunc})

    #add constraint for bottom hole and the hole below it
    hole = movingHoles[0]
    dx1 = dx.get(hole, totlength)
    dy1 = dy.get(hole, totlength)
    dr1 = dr.get(hole, totlength)

    #generate constraints, need normal and one that prevents loops
    constFunc = (lambda x,dx1=dx1,dy1=dy1,dr1=dr1: 
        npa.sqrt((np.append(x,0)[dx1]-.5)**2+
                 (np.append(x,0)[dy1]+np.sqrt(3)/2)**2)-
        2*ra-np.append(x,0)[dr1]-mindist)
    constraints.append({'type':'ineq','fun':constFunc})

    constFunc = (lambda x,dx1=dx1,dy1=dy1,dr1=dr1: 
        npa.sqrt((np.append(x,0)[dx1]+.5)**2+
                 (np.append(x,0)[dy1]+np.sqrt(3)/2)**2)-
        2*ra-np.append(x,0)[dr1]-mindist)
    constraints.append({'type':'ineq','fun':constFunc})

    #---------All spacial constraints are done--------

    #generate constraints for min and max frequency        
    #greater then the minimum frequency and less then the max frequency
    constFunc = lambda x: 6*80000*(-(costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][0]-(maxfreq-minfreq)/2-minfreq)**4+((maxfreq-minfreq)/2)**4)
    constraints.append({'type':'ineq','fun':constFunc})

    #we want to constrain the frequencys of the k-points before and after
    #start with k points before
    for i in range(len(kwargs["kpoints"][1])):
        if i==len(kwargs["kpoints"][1])-1:
            constFunc = lambda x,i=i: -costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][0]+costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][1][i]
        else:    
            constFunc = lambda x,i=i: -costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][1][i+1]+costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][1][i]
        constraints.append({'type':'ineq','fun':constFunc})

    #and then k points after
    for i in range(len(kwargs["kpoints"][2])):
        if i==0:
            constFunc = lambda x,i=i: costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][0]-costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][2][i]
        else:    
            constFunc = lambda x,i=i: costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][2][i-1]-costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][2][i]
        constraints.append({'type':'ineq','fun':constFunc})
    
    #now we want the v to be negative for all of them
    #start with at point
    constFunc = lambda x: -costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][1][0]
    constraints.append({'type':'ineq','fun':constFunc})
    
    #point before
    for i in range(len(kwargs["kpoints"][1])):
        constFunc = lambda x,i=i: -costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][1][1][i]
        constraints.append({'type':'ineq','fun':constFunc})

    #points after
    #and then k points after
    for i in range(len(kwargs["kpoints"][2])):
        constFunc = lambda x,i=i: -costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][1][2][i]
        constraints.append({'type':'ineq','fun':constFunc})


    #---------- all constraints on line done --------#
    #now we want to force band above and below outside of bandwidth
    for band in [3,6]:

        #for if the band is above or below
        pm = 1
        if band==6:
            pm = -1

        #start with at point
        constFunc = lambda x,pm=pm,band=band: pm*(-costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][0]+costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][band])-bandwidth
        constraints.append({'type':'ineq','fun':constFunc})
    
        #point before
        for i in range(len(kwargs["kpoints"][1])):
            constFunc = lambda x,pm=pm,band=band,i=i: pm*(-costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][0]+costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][band+1][i])-bandwidth
            constraints.append({'type':'ineq','fun':constFunc})

        #points after
        #and then k points after
        for i in range(len(kwargs["kpoints"][2])):
            constFunc = lambda x,pm=pm,band=band,i=i: pm*(-costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][0]+costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][band+2][i])-bandwidth
            constraints.append({'type':'ineq','fun':constFunc})
    return(constraints)
#%%
def TopoWavconstSingleMode(minfreq=0,maxfreq=1000,minrad=0,mindist=0,dx={},dy={},dr={},ra=(.235,.105),objective_function=of_Q,bandwidth=0,**kwargs):
    
    #set constraints initally empty
    constraints = []

    #get the combined length of all dictionaries, this is useful multiple times
    totlength = len(dx)+len(dy)+len(dr)

    #unpack ra
    r0,r1 = ra[0],ra[1]

    #generate the indexes that should be used for the constraint functions
    indexs = np.arange(totlength)
    dx,dy,dr = placeParams(indexs,dx=dx,dy=dy,dr=dr)

    #generate constraints for the radius 
    for key, i in dr.items():
        constFunc = lambda x,i=i: x[i]-minrad+npa.min([r1,r0])
        constraints.append({'type':'ineq','fun':constFunc})

    #generate constraints that hold the holes in their unit cell
    #this prevents the results from blowing up
    for key, i in dx.items():
        constFunc = lambda x,i=i: .5-npa.abs(x[i])
        constraints.append({'type':'ineq','fun':constFunc})
    for key, i in dy.items():
        constFunc = lambda x,i=i: .5-npa.abs(x[i])
        constraints.append({'type':'ineq','fun':constFunc})

    #we now generate constraints for holes that are close to eachother. we want one 
    #constraint per pair, and it should have constraints both between two moving holes 
    #and one moving hole and one statinoary hole
    #start by getting a list of all holes that that have parameters being altered
    movingHoles = []
    seenHoles = set()
    for dictionary in [dx,dy,dr]:
        for key, _ in dictionary.items():
            if key not in seenHoles:
                seenHoles.add(key)
                movingHoles.append(key)
    
    #sort moving holes so that they go form the bottom up
    movingHoles.sort(key=lambda x: (x[1], -x[2]))

    #We want to do nearest neighbor constarints. we only have to prevent the hole below
    #from interacting with the hole above it, but we need to do it such that it won't
    #loop around either
    for hole in movingHoles:

        #get the indexes for the variables we are changing in the hole we are focused on
        dx1 = dx.get(hole, totlength)
        dy1 = dy.get(hole, totlength)
        dr1 = dr.get(hole, totlength)

        #we have two cases: 0, when its the top hole, and 1 when its the bottom hole
        if hole[2]==0:

            #get the indexes for the variables we are changing in the hole above
            holeAbove = (hole[0],hole[1]+1,1)
            dx2 = dx.get(holeAbove, totlength)
            dy2 = dy.get(holeAbove, totlength)
            dr2 = dr.get(holeAbove, totlength)
        
            #the center hole is a double r1 hole, the rest will have one r1 and one r2
            if hole==(0,0,0):
                rtot = 2*r1
            else:
                rtot = r1+r0

            #generate constraints, need normal and one that prevents loops
            constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2,rtot=rtot: 
                npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2]-.5)**2+
                         (np.append(x,0)[dy1]-np.append(x,0)[dy2]-np.sqrt(3)/6)**2)-
                rtot-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
            constraints.append({'type':'ineq','fun':constFunc})

            constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2,rtot=rtot: 
                npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2]+.5)**2+
                         (np.append(x,0)[dy1]-np.append(x,0)[dy2]-np.sqrt(3)/6)**2)-
                rtot-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
            constraints.append({'type':'ineq','fun':constFunc})

        #second case, where hole is directly above
        elif hole[2]==1:

            #get the indexes for the variables we are changing in the hole above
            holeAbove = (hole[0],hole[1],0)
            dx2 = dx.get(holeAbove, totlength)
            dy2 = dy.get(holeAbove, totlength)
            dr2 = dr.get(holeAbove, totlength)

            #generate constraint func, we only need one case
            constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2: 
                npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2])**2+
                         (np.append(x,0)[dy1]-np.append(x,0)[dy2]-np.sqrt(3)/3)**2)-
                r0-r1-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
            constraints.append({'type':'ineq','fun':constFunc})
    

    #We need to set up the constraints for the bottom hole and the one below it
    #this is a lot of extra code, but it does the same thing as above
    hole = movingHoles[0]
    dx1 = dx.get(hole, totlength)
    dy1 = dy.get(hole, totlength)
    dr1 = dr.get(hole, totlength)
    

    #get the indexes for the variables we are changing in the hole below
    holeBelow = (hole[0],hole[1],1)
    dx2 = dx.get(holeBelow, totlength)
    dy2 = dy.get(holeBelow, totlength)
    dr2 = dr.get(holeBelow, totlength)

    #generate constraint func, we only need one case
    constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2: 
        npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2])**2+
                 (np.append(x,0)[dy1]-np.append(x,0)[dy2]+np.sqrt(3)/3)**2)-
        r0-r1-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
    constraints.append({'type':'ineq','fun':constFunc})

    #get the indexes for the variables we are changing in the hole below
    holeBelow = (hole[0],hole[1]-1,0)
    dx2 = dx.get(holeBelow, totlength)
    dy2 = dy.get(holeBelow, totlength)
    dr2 = dr.get(holeBelow, totlength)

    #generate constraints, need normal and one that prevents loops
    constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2: 
        npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2]-.5)**2+
                 (np.append(x,0)[dy1]-np.append(x,0)[dy2]+np.sqrt(3)/6)**2)-
        r0-r1-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
    constraints.append({'type':'ineq','fun':constFunc})

    constFunc = (lambda x,dx1=dx1,dx2=dx2,dy1=dy1,dy2=dy2,dr1=dr1,dr2=dr2: 
        npa.sqrt((np.append(x,0)[dx1]-np.append(x,0)[dx2]+.5)**2+
                 (np.append(x,0)[dy1]-np.append(x,0)[dy2]+np.sqrt(3)/6)**2)-
        r0-r1-np.append(x,0)[dr1]-np.append(x,0)[dr2]-mindist)
    constraints.append({'type':'ineq','fun':constFunc})

    #---------All spacial constraints are done--------

    #generate constraints for min and max frequency        
    #greater then the minimum frequency and less then the max frequency
    constFunc = lambda x: 6*80000*(-(costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][0]-(maxfreq-minfreq)/2-minfreq)**4+((maxfreq-minfreq)/2)**4)
    constraints.append({'type':'ineq','fun':constFunc})

    #we want to constrain the frequencys of the k-points before and after
    #start with k points before
    for i in range(len(kwargs["kpoints"][1])):
        if i==len(kwargs["kpoints"][1])-1:
            constFunc = lambda x,i=i: costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][0]-costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][1][i]
        else:    
            constFunc = lambda x,i=i: costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][1][i+1]-costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][1][i]
        constraints.append({'type':'ineq','fun':constFunc})

    #and then k points after
    for i in range(len(kwargs["kpoints"][2])):
        if i==0:
            constFunc = lambda x,i=i: -costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][0]+costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][2][i]
        else:    
            constFunc = lambda x,i=i: -costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][2][i-1]+costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][2][i]
        constraints.append({'type':'ineq','fun':constFunc})
    
    #now we want the v to be positive for all of them
    #start with at point
    constFunc = lambda x: costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][1][0]
    constraints.append({'type':'ineq','fun':constFunc})
    
    #point before
    for i in range(len(kwargs["kpoints"][1])):
        constFunc = lambda x,i=i: costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][1][1][i]
        constraints.append({'type':'ineq','fun':constFunc})

    #points after
    #and then k points after
    for i in range(len(kwargs["kpoints"][2])):
        constFunc = lambda x,i=i: costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][1][2][i]
        constraints.append({'type':'ineq','fun':constFunc})


    #---------- all constraints on line done --------#
    #now we want to force band above and below outside of bandwidth
    for band in [3,6]:

        #for if the band is above or below
        pm = 1
        if band==6:
            pm = -1

        #start with at point
        constFunc = lambda x,pm=pm,band=band: pm*(-costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][0]+costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][band])-bandwidth
        constraints.append({'type':'ineq','fun':constFunc})
    
        #point before
        for i in range(len(kwargs["kpoints"][1])):
            constFunc = lambda x,pm=pm,band=band,i=i: pm*(-costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][0]+costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][band+1][i])-bandwidth
            constraints.append({'type':'ineq','fun':constFunc})

        #points after
        #and then k points after
        for i in range(len(kwargs["kpoints"][2])):
            constFunc = lambda x,pm=pm,band=band,i=i: pm*(-costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][0]+costWrapper(x,returnFreq=True,objective_function=objective_function,dx=dx,dy=dy,dr=dr,ra=ra,**kwargs)[1][0][band+2][i])-bandwidth
            constraints.append({'type':'ineq','fun':constFunc})


    return(constraints)
