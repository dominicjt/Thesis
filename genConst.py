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

#%%

