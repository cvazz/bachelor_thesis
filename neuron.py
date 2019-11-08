import numpy as np
import math
import random 
from scipy import special
import scipy.integrate as integrate

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from scipy.stats import gaussian_kde

from pathlib import Path

import timeit
import time

import mathsim as msim


"""
older Connection Matrix Creator (28/10)

@param1     sizeM   : Contains size of exhib and inhib
@param2     jVal    : Contains nonzero Values for Matrix
@param3     K       : Number of connections with inhib/exhib

@return     jCon    : Connection Matrix
"""


def createjCon2(sizeM, jVal,K):
    sizeMax = sizeM[0] + sizeM[1]
    jCon = np.zeros((sizeMax,sizeMax))
    for index in np.ndindex(np.shape(jCon)):
        if index[0] < sizeM[0]:
            if index[1] <sizeM[0]:
                if random.uniform(0,1)<K/sizeM[0]:
                    jCon[index] = jVal[0,0]
            else:
                if random.uniform(0,1)<K/sizeM[1]:
                    jCon[index] = jVal[0,1]
        else:
            if index[1] <sizeM[0]:
                if random.uniform(0,1)<K/sizeM[0]:
                    jCon[index] = jVal[1,0]
            else:
                if random.uniform(0,1)<K/sizeM[1]:
                    jCon[index] = jVal[1,1]
    print("Save to File to decrease runtimes")
    print("Try with np.random.bernoulli ")
    return jCon

"""
Current Connection Matrix Creator (31/10)

creates ordered list of just ones and then just zeros. This is then shuffled by using sample(len(obj)) with obj being the object to shuffle

@param1     sizeM   : Contains size of exhib and inhib
@param2     jVal    : Contains nonzero Values for Matrix
@param3     K       : Number of connections with inhib/exhib

@return     jCon    : Connection Matrix
"""
def createjCon3(sizeM, jVal,K):
    debug = 0
    np.set_printoptions(edgeitems = 5)
    sizeMax = sizeM[0] + sizeM[1]
    #distribute Connections
    ones = int(2*K*sizeMax)
    if debug: print(ones)
    jCon = [0] * (sizeMax**2 -ones) + [1] * ones 
    if debug: print(np.array(jCon))
    jCon = random.sample(jCon,len(jCon))
    if debug: print(np.array(jCon))

    #shuffle
    jCon = np.array(jCon, dtype = float)
    jCon.shape = (sizeMax,sizeMax)

    #add weights
    jCon[:sizeM[0],:sizeM[0]] = np.multiply(jCon[:sizeM[0],:sizeM[0]],jVal[0,0])
    jCon[sizeM[0]:,:sizeM[0]] = jCon[sizeM[0]:,:sizeM[0]]*jVal[1,0]
    jCon[:sizeM[0],sizeM[0]:] = jCon[:sizeM[0],sizeM[0]:]*jVal[0,1]
    jCon[sizeM[0]:,sizeM[0]:] = jCon[sizeM[0]:,sizeM[0]:]*jVal[1,1]

    return jCon


def createNval(sizeM, extM, K, mean0):
    nval = []
    ones = mean0 * sizeM
    for i in range(len(sizeM)):
        numof1 = int(ones[i])
        numof0 = sizeM[i] - numof1
        arr = [0] * numof0 + [1] * numof1
        arr = random.sample(arr,len(arr))
        nval+= arr
    return np.array(nval)

"""
Creates Thresholds for each Datapoint
Currently uniformly for each group
"""
def createThresh(sizeM, threshM):
    thresh= []
    for i in range(2):
        thresh.extend([threshM[i] for x in range(sizeM[i])])
    return np.array(thresh)


"""
Not sure whether still needed
Gaussian Distribution for variation in thresholds
"""
def createAltThresh(nval):
    mean  = 1. 
    dev = 0.2
    thresh = np.empty_like(nval,dtype=float)
    for index in np.ndindex(np.shape(thresh)):
        thresh[index] =  np.random.normal(mean,dev)
    return thresh

"""

"""
def createExt(sizeM, extM, K, mean0):
    ext = []
    extVal = extM * math.sqrt(K) *mean0
    for i in range(len(sizeM)):
        ext.extend([extVal[i] for x in range(sizeM[i])])
    return np.array(ext)


def createExtAlt(extM, sizeM):
    return np.array([extM[0] for _ in range(sizeM[0])] + 
                        [extM[1] for _ in range(sizeM[1])] )

"""
Current Calculator for whether one neuron changes value or not (31/10)

Sums all the input with corresponding weights. 
Afterwords adds external input and subtracts threshold. 
Result is plugged in Heaviside function

@param      iter    : iterator, determines which neuron is to be changed
@param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
@param      jCon    : Connection Matrix 
@param      thresh  : Stores Thresholds 
@param      external: Input from external Neurons 
@param      fire    : records how often a neuron switches to active state 

@return         : for troubleshooting returns value before Heaviside function
"""
def timestepMat (iter, nval, jCon, thresh, external, fire):
    debug = 0
    sum = jCon[iter].dot(nval)
    decide = sum + external[iter] - thresh[iter]
    if debug: print("iter:\t"+ str(iter) + "\tdecide:\t" + str(decide))
    if sum + external[iter] - thresh[iter] > 0:
        if nval[iter] == 0:
            fire[iter] += 1
        nval[iter] = 1
    else:
        nval[iter] = 0
    return sum + external[iter] - thresh[iter]

def timestepMatRecord(iter, nval, jCon, thresh, external, fire,sizeM):
"""
Current Calculator for whether one neuron changes value or not(31/10)

Particularity: Records additional information ie positive and negative input 
Sums all the input with corresponding weights. 
Afterwords adds external input and subtracts threshold. 
Result is plugged in Heaviside function

@param      iter    : iterator, determines which neuron is to be changed
@param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
@param      jCon    : Connection Matrix 
@param      thresh  : Stores Thresholds 
@param      external: Input from external Neurons 
@param      fire    : records how often a neuron switches to active state 

@return         : returns positive input from other neurons, all input (-threshold) and negative input
"""
    debug = 0

    pos = jCon[iter,:sizeM[0]].dot(nval[:sizeM[0]])
    if debug: print("pos")
    if debug: print(pos)
    neg = jCon[iter,sizeM[0]:].dot(nval[sizeM[0]:])
    if debug: print("neg")
    if debug: print(neg)

    summe = pos + neg
    sum0 = jCon[iter].dot(nval)
    if abs(sum0-summe) > 0.001:
        print("Ungenauigkeit durch Messugn")
    decide =summe + external[iter] - thresh[iter]
    if debug: print("decide")
    if debug: print(decide)
    if summe + external[iter] - thresh[iter] > 0:
        if nval[iter] == 0:
            fire[iter] += 1
        nval[iter] = 1
    else:
        nval[iter] = 0
    return [pos,summe + external[iter],neg]

"""
Updates neurons in the sequence 1 to N

if smaller than recNum (or closer to sizeMax for inhibitory) then all changes are recorded in recorder
@param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
@param      jCon    : Connection Matrix 
@param      thresh  : Stores Thresholds 
@param      external: Input from external Neurons 
@param      fire    : records how often a neuron switches to active state CHANGES
@param      time    : Controls runtime
@param      sizeM   : Contains information over the network size
@param      recorder:  CHANGES
@param      recNum  : How many neurons are recorded 

@return    Nothing 
"""
def sequential(nval, jCon,  thresh, external, fire, recorder, recNum, sizeM):
    debug = 0
    excite = 1
    sizeMax = sum(sizeM)
    
    for iter in range(len(nval)):
        if excite:
            if iter <recNum:
                recorder[iter].append(timestepMatRecord(
                    iter, nval, jCon, thresh, external, fire, sizeM))
            else:
                timestepMat (iter, nval, jCon, thresh, external, fire)
        else:
            if iter > sizeMax -recNum:
                recorder[sizeMax - iter-1].append(timestepMatRecord(
                    iter, nval, jCon, thresh, external, fire, sizeM))
            else:
                timestepMat (iter, nval, jCon, thresh, external, fire)

    if debug == 1:
        print (nval)
"""
Runs the sequence 1 to N for "time" times. 

Manages Recordings aswell (see @return)

@param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
@param      jCon    : Connection Matrix 
@param      thresh  : Stores Thresholds 
@param      external: Input from external Neurons 
@param      fire    : records how often a neuron switches to active state 
@param      time    : Controls runtime
@param      sizeM   : Contains information over the network size
@param      recNum  : How many neurons are recorded 

@return     Returns recorder and total which analyze individual (ie subset of) neurons and all neurons respectively,
            aswell as recordEverything which records nval after every time step
"""
def sequRun(jCon, thresh, external, timer, sizeM, extM, K, mean0, recNum = 10):
    debug = 1
    nval = createNval(sizeM, extM, K, mean0)  
    fire = np.zeros(np.shape(nval)) #Fire rate for indivual neuron
    total = np.zeros_like(nval)
    recordEverything = [total.copy() for x in range(timer)]
    recorder = [[] for i in range(recNum)] 
    timediff = []
    for t in range(0,timer):
        timestart = time.time()
        sequential(nval, jCon, thresh, external, fire, recorder, recNum, sizeM)
        if timer>50:
            if t%50==0:
                print(t)
        total += (nval)
        recordEverything[t] += nval
        timeend = time.time()
        timediff.append(timeend - timestart)
    print("mean time for one cycle")
    timeOut(np.mean(np.array(timediff)))
    return recorder, total, fire, recordEverything

"""
Randomly chooses between excitatory or inhibitory sequence

Randomly chooses between excitatory or inhibitory sequence with relative likelihood tau 
to choose inhibitory (ie 1 meaning equally likely).
Currently only supports recording individual excitatory neurons


@param      timeOut : Controls runtime
@param      sizeM   : Contains information over the network size
@param      tau     : How often inhibitory neurons fire compared to excitatory
@param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
@param      jCon    : Connection Matrix 
@param      thresh  : Stores Thresholds 
@param      external: Input from external Neurons 
@param      fire    : records how often a neuron switches to active state 
@param      recorder: 
@param      recNum  : How many neurons are recorded 

@return     Nothing
"""
def poissonish(sizeM,timeOut, tau, nval, jCon, thresh, external, fire, recorder, recNum):
    total = np.zeros_like(nval)
    sizeMax = sizeM[0] + sizeM[1]
    prob =  1. / (1+tau)
    exTime = 0
    inTime = 0
    exIterStart = 0
    inIterStart = sizeM[0]
    exIter = exIterStart 
    inIter = inIterStart 
    recordEverything = np.zeros((sum(sizeM),timeOut))
    while exTime < timeOut:
        if np.random.uniform(0,1)<prob:
            if exIter <recNum:
                vals = timestepMatRecord(exIter, nval, jCon,
                thresh, external, fire, sizeM)
                recorder[exIter].append(vals)
                if vals[1] >= 0:
                    recordEverything[exIter,exTime]
            else:
                overThresh = timestepMat (exIter, nval, jCon, thresh, external, fire)
                if overThresh >= 0:
                    recordEverything[exIter,exTime]
            exIter +=1
            if exIter == inIterStart:
                exTime+=1
                exIter = exIterStart
                total = nval[:sizeM[0]]
                if exTime % 10 == 0:
                    print(exTime)
        else:
            overThresh = timestepMat (inIter, nval, jCon, thresh, external, fire)
            if overThresh >= 0:
                recordEverything[inIter,inTime]
            inIter += 1
            if inIter == sizeMax:
                inTime+=1
                inIter = inIterStart
                total = nval[sizeM[0]:]
    return recordEverything
"""
Runs a mix between poisson and sequential Code

Introduces analyze tools


@param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
@param      jCon    : Connection Matrix 
@param      thresh  : Stores Thresholds 
@param      external: Input from external Neurons 
@param      fire    : records how often a neuron switches to active state 
@param      timeOut : Controls runtime
@param      sizeM   : Contains information over the network size
@param      tau     : How often inhibitory neurons fire compared to excitatory
@param      recNum  : How many neurons are recorded 

@return     Returns recorder and total which analyze individual (ie subset of) neurons and all neurons respectively

"""
def poissRun(jCon, thresh, external, timeOut,
    sizeM, extM, K, mean0, tau, recNum = 10):

    nval = createNval(sizeM, extM, K, mean0)  
    fire = np.zeros(np.shape(nval)) #Fire rate for indivual neuron
    total = np.zeros_like(nval)
    recorder = [[] for i in range(recNum)] 

    recordEverything = poissonish(sizeM,timeOut, tau, nval, jCon, thresh, external, fire, recorder, recNum)
    total = fire
    return recorder, total, fire, recordEverything
"""
Utitlity Function:
loads built Connection Matrices into process for speed gains
"""
def createjConFromFile(filename):
    jCon = np.load(filename)
    return jCon

"""
Utility Function: To save runtimes during testing, connection matrices are saved and reloaded if specificatinos match

"""
def jConTinker(sizeM, jVal,K):
    filepath =  Path("/mnt/c/Users/sebas/Desktop/saveLocation")
    if not filepath.exists():
        filepath =  Path("/mnt/c/Users/Sebaschdiaan/Desktop/saveLocation")
    if sizeM[0] == 10000:
        loc = filepath /'test10up5.npy'
    elif sizeM[0] == 1000:
        loc = filepath /'test10up4.npy'
    elif sizeM[0] == 2000:
        loc = filepath /'test20up4.npy'
    elif sizeM[0] == 20000:
        loc = filepath /'test20up5.npy'
    if 'loc' in locals():
        if loc.exists():
            jCon = createjConFromFile(loc)
        else:
            jCon = createjCon3(sizeM, jVal,K)
    else:
        jCon = createjCon3(sizeM, jVal,K)
    if 'loc' in locals():
        if not loc.exists():
            np.save(loc, jCon)
            pass
    return jCon


def testthename(name,fileEnding):
    workinName = name + "." + fileEnding
    pathname = Path(workinName)
    count = 0
    while pathname.exists():
        count += 1
        workinName = name +"_no_" +str(count) + "." + fileEnding
        pathname = Path(workinName)
    return workinName

def checkFolder(foldername):
    folderpath = Path(foldername)
    if not folderpath.exists():
        folderpath.mkdir()
    return foldername + "/"

def relMax(fire,showRange):
    return np.argpartition(fire, -1*showRange)[-1*showRange:]
"""
plots distribution of firing pattern in relation to mean firing pattern

(currently (everything larger than 5*mean is labelled as 6)
@param      total   : contains all the times a specific neuron fired
@param      fire    : contains all the times a specific neuron spiked (ie turned 0 afterwards) (not in use)
"""
def plotTotal(foldername, total, fire, timer, titletxt, captiontxt):

    total = np.array(total)
    meanTot = np.mean(total)
    if meanTot == 0:
        print("not a single flip for the following starting values")
        txt = f'Network Size: {sizeMax} \t K: {K} \t mean_0: {mean0} \n total time: {timer}   jE: {jE}\t jI: {jI}\t '
        print(txt)
        return 
    total = total/meanTot
    #total2 = [6 if tot>5 else tot for tot in total ]
    density = gaussian_kde(total)
    xs = np.linspace(0,3)
    density.covariance_factor = lambda : .1
    density._compute_covariance()
    fig = plt.figure()
    plt.plot(xs,density(xs))

    plt.title('Fire Rate Distribution')
    plt.xlabel('Fire rate/mean')
    plt.ylabel('Density')
    fig.text(.5,.05,captiontxt, ha='center')
    fig.subplots_adjust(bottom=0.2)

    folder = checkFolder(foldername)
    name = "density"
    fullname = testthename(folder +name+titletxt , "png")
    plt.savefig(fullname)
    #plt.show()
    plt.close(fig)
    
    histfig = plt.figure()
    uniq = len(np.unique(total))
    binsize = 10 if uniq <10 else uniq if uniq<timer else timer
    plt.hist(total, bins = binsize)
    plt.title('Histogram of Fire Rate Distribution')
    plt.xlabel('fire rate/mean')
    plt.ylabel('density')
    histfig.text(.5,.05,captiontxt, ha='center')
    histfig.subplots_adjust(bottom=0.2)

    folder = checkFolder(foldername)
    name = "histogram"
    fullname = testthename(folder +name+titletxt , "png")
    plt.savefig(fullname)
    #plt.show()
    plt.close(histfig)

"""
Plots inputs in several neurons (ie 3 to 10)

Shows positive, negative and total input for several neurons 

@param      recorder     :Contains pos, neg, and total value for a subgroup of neurons at each timestep

"""
def plotIndi(foldername, recorder, fire, threshM, titletxt, captiontxt):
    showRange = 5
    exORin = 0
    fig = plt.figure()
    recSize = len(recorder)
    showTheseRows = relMax(fire[:recSize],showRange)
    for i in showTheseRows:
        rec = np.array(recorder[i])
        rec = np.transpose(rec)
        xs = range(0,len(rec[1]))
        consta = [threshM[exORin] for x in range(len(rec[1]))]
        col=['red', 'green', 'blue']
        for j in range(len(rec)):
            plt.plot(xs,rec[j], color = col[j], linewidth = .8)

    plt.plot(xs,consta, color = "black", linewidth = 2.0)
    fig.text(.5,.05,captiontxt, ha='center')
    fig.subplots_adjust(bottom=0.2)
    plt.title('Individual Neuron Firing Pattern')
    plt.xlabel('time')
    plt.ylabel('Current')

    folder = checkFolder(foldername)
    name = "Indi"
    fullname = testthename(folder +name+titletxt , "png")
    plt.savefig(fullname)
    #plt.show()
    plt.close(fig)


"""
Plots inputs in several neurons (ie 3 to 10)

Shows positive, negative and total input for several neurons aswell as when they spike

@param      recorder     :Contains pos, neg, and total value for a subgroup of neurons at each timestep

"""
def plotIndiAlt(foldername, recorder, fire, threshM, titletxt, captiontxt):
    showRange = 10
    recNum = 1000
    exORin = 0
    #gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    for level in range(int(recNum/showRange)):
        fig = plt.figure()
        if level % 10 == 0:
            print(level)
        level *= 10
        for i in range(level, showRange + level):
            rec = np.array(recorder[i])
            rec = np.transpose(rec)
            xs = range(0,len(rec[1]))
            consta = [threshM[exORin] for x in range(len(rec[1]))]
            col=['red', 'green', 'blue']
            for j in range(len(rec)):
                plt.plot(xs,rec[j], color = col[j], linewidth = .8)

        plt.plot(xs,consta, color = "black", linewidth = 2.0)
        fig.text(.5,.05,captiontxt, ha='center')
        fig.subplots_adjust(bottom=0.2)
        plt.title('Individual Neuron Firing Pattern')
        plt.xlabel('time')
        plt.ylabel('Current')

        folder = checkFolder(foldername)
        name = "Indi"
        fullname = testthename(folder +name+titletxt , "png")
        plt.savefig(fullname)
        #plt.show()
        plt.close(fig)
def plotIndiExtended(foldername, recorder, fire, threshM, titletxt, captiontxt):
    showRange = 15
    recNum = 1000
    exORin = 0
    level = 0
    fig, axarr = plt.subplots(2,sharex=True,)
    dataspike = []
    #gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    for i in range(level, showRange + level):
        rec = np.array(recorder[i])
        rec = np.transpose(rec)
        xs = range(0,len(rec[1]))
        consta = [threshM[exORin] for x in range(len(rec[1]))]
        col=['red', 'green', 'blue']
        spike = [1 if rec[1][j] > 1 else 0 for j in range(len(rec[1]))]
        dataspike.append(spike)
        for j in range(len(rec)):
            axarr[0].plot(xs,rec[j], color = col[j], linewidth = .8)
    axarr[1].imshow(dataspike, aspect='auto', cmap='Greys', interpolation='nearest')

    axarr[0].plot(xs,consta, color = "black", linewidth = 2.0)
    fig.text(.5,.05,captiontxt, ha='center')
    fig.subplots_adjust(bottom=0.2)
    plt.title('Individual Neuron Firing Pattern')
    plt.xlabel('time')
    axarr[0].set(ylabel = 'Current')
    axarr[1].set(ylabel = 'Spike')

    folder = checkFolder(foldername)
    name = "IndiExt"
    fullname = testthename(folder +name+titletxt , "png")
    plt.savefig(fullname)
    #plt.show()

def plottau(foldername, recordEverything, timer, titletxt, captiontxt):
    diff = []
    for rec in np.transpose(recordEverything):
        dist = [x for x in range(len(rec)) if rec[x] == 1 ]
        diff += [dist[i+1]-dist[i] for i in range(len(dist)-1)]


    histfig = plt.figure()
    uniq = len(np.unique(diff))
    binsize = 10 if uniq <10 else uniq if uniq<timer else timer
    plt.hist(diff, bins = binsize)
    plt.title('Histogram of Fire Intervals')
    plt.xlabel('time between firing two times')
    plt.ylabel('density')
    histfig.text(.5,.05,captiontxt, ha='center')
    histfig.subplots_adjust(bottom=0.2)


    folder = checkFolder(foldername)
    name = "intervalHist"
    fullname = testthename(folder +name+titletxt , "png")
    plt.savefig(fullname)
    #plt.show()
    plt.close(histfig)


def analyzeTau(rec):
    dist =[]
    buff = 1
    counter = 0
    while rec[0] == 0:
        rec.pop(0)
        if not rec:
            return 0
    if not rec:
        rec.pop(0)
    for i in range(len(rec)):
        if rec[i] == 0:
            counter += 1
            buff = 0
        else: #elif rec[i] == 1:
            if buff == 0:
                dist.append(counter)
                counter = 0
                buff = 1
    return dist
def plottau2(foldername, recordEverything, timer, titletxt, captiontxt):

    diff = []
    for rec in np.transpose(recordEverything):
        a = analyzeTau(rec.tolist())
        if a: 
            diff += a


    histfig = plt.figure()
    uniq = len(np.unique(diff))
    binsize = 10 if uniq <10 else uniq if uniq<timer else timer
    plt.hist(diff, bins = binsize)
    plt.title('Histogram of Fire Intervals V2')
    plt.xlabel('time between firing two times')
    plt.ylabel('density')
    histfig.text(.5,.05,captiontxt, ha='center')
    histfig.subplots_adjust(bottom=0.2)


    folder = checkFolder(foldername)
    name = "intervalHist2"
    fullname = testthename(folder +name+titletxt , "png")
    plt.savefig(fullname)
    #plt.show()
    plt.close(histfig)

def plotDots(foldername, recordEverything, timer, titletxt, captiontxt):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    record =np.transpose(recordEverything)
    ax.imshow(record, aspect='auto', cmap='Greys', interpolation='nearest')
    plt.title('Histogram of Fire Intervals V2')
    plt.xlabel('time between firing two times')
    plt.ylabel('density')
    fig.text(.5,.05,captiontxt, ha='center')
    fig.subplots_adjust(bottom=0.2)


    folder = checkFolder(foldername)
    name = "dots"
    fullname = testthename(folder +name+titletxt , "png")
    plt.savefig(fullname)
    #plt.show()
    plt.close(fig)

"""
creates all the needed objects and calls the workload functions and plots.
Virtually a main function, without parameter definition
"""
def prepare(K, mean0, tau, sizeE, sizeI, extE, extI, jE, jI, threshE, threshI):

    debug = 0
    ### SIZE ###
    sizeM = np.array([sizeE, sizeI])
    sizeMax = sizeE + sizeI
    sizeM.setflags(write=False)

    if min(sizeM)<K:
        raise ValueError("K must be smaller than or equal to size "
                        +"of excitatory or inhibitory Values")
    ### External Input ###
    extM = np.array([extE, extI])
    extM.setflags(write=False)
    external = createExt(sizeM,extM, K, mean0)     
    external.setflags(write=False)

    ### Threshoold Level ###
    threshM = np.array([threshE, threshI])
    threshM.setflags(write=False)
    thresh = createThresh(sizeM, threshM) #createAltThresh generates gaussian
    thresh.setflags(write=False)

    ### Values of Connection Matrix ###
    jEE = 1
    jIE = 1
    jEI = -1*jE
    jII = -1*jI
    jVal = np.array([[jEE, jEI],[jIE, jII]])
    jVal = jVal/math.sqrt(K)

    ### Connection Matrix ###
    print("Create jCon")
    timestart = time.time()
    jCon = jConTinker(sizeM, jVal,K)
    jCon.setflags(write=False)
    timeend = time.time()
    timeOut(timeend - timestart)
    return sizeM, threshM, extM, jCon, thresh, external

def timeOut(timediff):
    if timediff>200:
        mins = int(timediff/60)
    else: 
        mins = 0
    if mins> 100:
        hours = int (mins/60)
    print('{:4.3} s'.format(timediff))

def testRoutine(foldername, timer, K, mean0, tau, sizeM,threshM, extM,
    jCon, thresh, external, jE, jI, extratxt = "",
    pTot=0, pIndi=0, pIndiExt= 0, pDist=0, pDist2=0, pDots=0):

    sizeMax = sizeM[0] + sizeM[1]
    np.set_printoptions(edgeitems = 10)
    titletxt = f'S_{int(np.log10(sizeMax))}_K_{(K)}_m0_{mean0}'# \njE: {jE}   jI: {jI} '
    titletxt = f's_{float(sizeMax):2.0}_K_{(K)}_m0_{int(np.log10(mean0))}'# \nje: {je}   ji: {ji} '
    captiontxt = f'Network Size: {sizeMax}  K: {K}  mean_0: {mean0} \n\
        total time: {timer}   jE: {jE}   jI: {jI} ' + extratxt

    recNum = 1000
    print("run")
    timestart = time.time()
    #recorder, total, fire, recordEverything = sequRun(jCon, thresh, external, timer ,sizeM, extM, K, mean0, recNum)
    recorder,total, fire, recordEverything= poissRun(jCon, thresh, external, timer, sizeM, extM, K, mean0, tau)
    timeend = time.time()
    print("runtime of routine")
    timeOut(timeend - timestart)

    ### Plotting Routine ###
    if pTot:
        try:
            plotTotal(foldername, total,fire, timer, titletxt, captiontxt)
        except ValueError:
            print("ValueError at total, most likely no 0s")
            print(total)
    if pIndi:
        plotIndi(foldername,recorder,fire, threshM, titletxt, captiontxt)
    if pIndiExt:
        plotIndiExtended(foldername,recorder,fire, threshM, titletxt, captiontxt)
    if pDist:
        plottau(foldername, recordEverything, timer, titletxt, captiontxt)
    if pDist2:
        plottau2(foldername, recordEverything, timer, titletxt, captiontxt)
    plotDots(foldername, recordEverything, timer, titletxt, captiontxt)




def main():
    timestr = time.strftime("%y%m%d_%H%M")
    foldername = "figs/testreihe" + timestr
    """
    folderpath = Path(foldername)
    if not folderpath.exists():
        folderpath.mkdir()
    foldername +=  "/"
    """    
    timer   = 200
    sizeE   = 10000
    sizeI   = 10000
    extE    = 1.
    extI    = 0.8
    jE      = 2.
    jI      = 1.8
    threshE = 1.
    threshI = 0.7
    tau     = 0.9
    mean0   = 0.1
    K       = 1000

    ### Deviations ###
    size    = 100
    sizeE   = size
    sizeI   = size
    K       = 100
    mean0   = 0.1

    #backup_correctparameters()
    #troubleshootParamters()

    #Bools for Wh
    pTot    = 1
    pIndi   = 0
    pIndiExt= 1
    pDist   = 1
    pDist2  = 1
    pDots   = 1
    
    print("np.binomial(size) -> shape -> fertig")
    print("permutation")
    print("Ergebnisse abspeichern und danach plotten")
    print("What exactly is Poisson statistics and process")
    extratxt = ""
    
    sizeM, threshM, extM, jCon, thresh, external = prepare(
        K, mean0, tau, sizeE, sizeI, extE, extI, jE, jI,
        threshE, threshI)

    testRoutine(
        foldername, timer, K, mean0, tau, sizeM, threshM, extM,
        jCon, thresh, external,jE, jI, extratxt,
        pTot, pIndi, pIndiExt, pDist,pDist2,pDots)

    external = createExtAlt(extM, sizeM)
    extratxt = "alternative extM"
    testRoutine(
        foldername, timer, K, mean0, tau, sizeM, threshM, extM,
        jCon, thresh, external,jE, jI, extratxt,
        pTot, pIndi, pIndiExt, pDist,pDist2,pDots)
    """
    for jE in jEXX:
        for jI in jIXX:
            testRoutine(foldername, timer, K, mean0, tau, sizeM, extM, jCon, thresh, external)
            pass
    #"""
#Link: About replacing part of array
# https://stackoverflow.com/questions/26506204/replace-sub-part-of-matrix-by-another-small-matrix-in-numpy

if __name__ == "__main__":
    main()
