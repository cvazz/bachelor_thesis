"""
Current File containing all relevant functions for my bachelor thesis
"""
import numpy as np
import math
import random 
from scipy import special
import scipy.integrate as integrate

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from collections import OrderedDict #grouped Labels

from pathlib import Path
import pickle

import time

import mathsim as msim
import utils
import plots


################################################################################
############################## Global Variables ################################
################################################################################
load_jConMatrix_from_can_GLOBAL = 0
plotLoc_print2console_GLOBAL = 1
################################################################################
############################## Utility Functions ###############################
################################################################################
def createjConFromFile(filename):
    """
    Utitlity Function:

    loads built Connection Matrices into process for speed gains
    """
    jCon = np.load(filename, allow_pickle = True)
    return jCon


def jConTinker(sizeM, jVal,K):
    """
    Utility Function: To save runtimes
    
     during testing, connection matrices are saved
     and reloaded if specificatinos match

    """
    folderCon =  Path("jCon_Repository")
    if not folderCon.exists():
        folderCon.mkdir()
    if not load_jConMatrix_from_can_GLOBAL:
        pass
    elif sizeM[0] == 10000:
        loc = folderCon /'test10up5.npy'
    elif sizeM[0] == 1000:
        loc = folderCon /'test10up4.npy'
    elif sizeM[0] == 2000:
        loc = folderCon /'test20up4.npy'
    elif sizeM[0] == 20000:
        loc = folderCon /'test20up5.npy'
    if 'loc' in locals():
        if loc.exists():
            jCon = createjConFromFile(loc)
        else:
            jCon = createjCon(sizeM, jVal,K)
    else:
        jCon = createjCon(sizeM, jVal,K)
    if 'loc' in locals():
        if not loc.exists():
            np.save(loc, jCon)
            pass
    return jCon

def saveResults(valueFolder, indiNeuronsDetailed, fireCount, nval_over_time, 
                timer, threshM, titletxt, captiontxt):
    """
    Save recordings of output to file.
    
    :param valueFolder: Path to storage folder
    :type valueFolder: class: 'pathlib.PosixPath'
    :param indiNeuronsDetailed:  

    """
    if not valueFolder.exists():
        valueFolder.mkdir(parents = True)

    infoDict = {
        "timer"     : timer, 
        "threshM"   : threshM,
        "titletxt"  : titletxt,
        "captiontxt": captiontxt,
        }
    infoDict["nvalOT_shape"] = np.shape(nval_over_time)

    loc = []
    uniq = np.unique(nval_over_time)
    if len(uniq) > 2:
        print(f"Warning nval_over_time has recorded non boolean values,"+
        " which have not been saved")
        print (np.unique(nval_over_time, return_counts=1))
    for iter in np.ndindex(np.shape(nval_over_time)):
        if nval_over_time[iter]:
            loc.append(iter)
    loc = np.array(loc)
    #loop next time
    indiNametxt     = "indiNeurons"
    fireNametxt     = "fireCount"
    #nvalOTNametxt   = "nval_OT"
    nvalIndexNametxt   = "nval_Index"
    infoNametxt     = "infoDict"

    indiName        = utils.makeNewPath(valueFolder, indiNametxt, "npy")
    fireName        = utils.makeNewPath(valueFolder, fireNametxt, "npy")
    #nvalOTName      = utils.makeNewPath(valueFolder, nvalOTNametxt, "npy")
    nval_Index_Name = utils.makeNewPath(valueFolder, nvalIndexNametxt, "npy")
    infoName        = utils.makeNewPath(valueFolder, infoNametxt, "pkl")

    np.save(indiName, indiNeuronsDetailed)
    np.save(fireName, fireCount)
    #np.save(nvalOTName, nval_over_time)
    np.save(nval_Index_Name, loc)
    infoName.touch()
    with open(infoName, "wb") as infoFile:
        pickle.dump(infoDict, infoFile, protocol = pickle.HIGHEST_PROTOCOL)

def recoverResults(valueFolder):
    """
    Load saved results from file.

    Missing functionality: so far, does only take neurons with preset names

    :param valueFolder: |valueFolder_desc|
    :return: indiNeuronsDetailed, fireCount, 
    """

    indiNametxt = "indiNeurons"
    fireNametxt = "fireCount"
    #nvalOTNametxt = "nval_OT"
    nvalIndexNametxt   = "nval_Index"
    infoNametxt     = "infoDict"

    indiName        = utils.makeExistingPath(valueFolder, indiNametxt, "npy")
    fireName        = utils.makeExistingPath(valueFolder, fireNametxt, "npy")
    #nvalOTName      = utils.makeExistingPath(valueFolder, nvalOTNametxt, "npy")
    nval_Index_Name = utils.makeExistingPath(valueFolder, nvalIndexNametxt, "npy")
    infoName        = utils.makeExistingPath(valueFolder, infoNametxt, "pkl")

    indiNeuronsDetailed = np.load(indiName, allow_pickle = True)
    fireCount           = np.load(fireName, allow_pickle = True)
    #nval_over_time      = np.load(nvalOTName, allow_pickle = True)
    nval_Index          = np.load(nval_Index_Name, allow_pickle = True)

    with open(infoName, 'rb') as infoFile:
        infoDict = pickle.load(infoFile)
    nval_OT = np.zeros(infoDict["nvalOT_shape"])
    for it in nval_Index:
        iter = (it[0], it[1])
        nval_OT[iter] = 1

    return indiNeuronsDetailed, fireCount, nval_OT, infoDict

################################################################################
############################ Creating Functions ################################
################################################################################
def createjCon(sizeM, jVal,K):
    """
    Current Connection Matrix Creator (31/10)

    Only working for 

    :param     sizeM   : Contains size of exhib and inhib
    :param     jVal    : Contains nonzero Values for Matrix
    :param     K       : Number of connections with inhib/exhib

    :return     jCon    : Connection Matrix
    """
    if sizeM[0] != sizeM[1]:
        raise ValueError("probability assumes equal likelihood of being excitatory or inhibitory")

    debug       = 0
    sizeMax     = sizeM[0] + sizeM[1]

    oddsBeingOne= 2*K/sizeMax
    jCon        = np.random.binomial(1, oddsBeingOne, sizeMax**2)

    jCon        = jCon.astype(float)
    jCon.shape  = (sizeMax,sizeMax)

    #add weights
    jCon[:sizeM[0],:sizeM[0]] = np.multiply(jCon[:sizeM[0],:sizeM[0]],jVal[0,0])
    jCon[sizeM[0]:,:sizeM[0]] = jCon[sizeM[0]:,:sizeM[0]]*jVal[1,0]
    jCon[:sizeM[0],sizeM[0]:] = jCon[:sizeM[0],sizeM[0]:]*jVal[0,1]
    jCon[sizeM[0]:,sizeM[0]:] = jCon[sizeM[0]:,sizeM[0]:]*jVal[1,1]

    return jCon


def createNval(sizeM, extM, K, mean0):
    """
    Initializes neuron values "nval" with starting values

    :param      sizeM   : Contains size of exhib and inhib neurons
    :param      extM    : Contains factors of external neurons for the inhib values in the system
    :param      K       : Connection Number
    :param      mean0   : Mean activation of external neurons
    """
    nval = []
    ones = mean0 * sizeM
    for i in range(len(sizeM)):
        numof1 = int(ones[i])
        numof0 = sizeM[i] - numof1
        arr = [0] * numof0 + [1] * numof1
        arr = random.sample(arr,len(arr))
        nval+= arr
    return np.array(nval)

def createThresh(sizeM, threshM):
    """
    Creates Threshold vector with threshold for each Datapoint

    :param      sizeM   : Contains size of exhib and inhib neurons
    :param      threshM : Contains values for threshold
    """
    thresh= []
    for i in range(2):
        thresh.extend([threshM[i] for x in range(sizeM[i])])
    return np.array(thresh)



def createExt(sizeM, extM, K, mean0):
    """
    Creates vector of external input for each Datapoint 
    
    (with all exhib and all inhib having the same value)

    :param      sizeM   : Contains size of exhib and inhib neurons
    :param      extM    : Contains factors of external neurons for the inhib values in the system
    :param      K       : Connection Number
    :param      mean0   : Mean activation of external neurons
    """
    ext = []
    extVal = extM * math.sqrt(K) *mean0
    for i in range(len(sizeM)):
        ext.extend([extVal[i] for x in range(sizeM[i])])
    return np.array(ext)


################################################################################
############################## Core Functions ##################################
################################################################################
def timestepMat (iter, nval, jCon, thresh, external, fireCount):
    """
    Calculator for whether one neuron changes value

    Sums all the input with corresponding weights. 
    Afterwords adds external input and subtracts threshold. 
    Result is plugged in Heaviside function

    :param      iter    : iterator, determines which neuron is to be changed
    :param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
    :param      jCon    : Connection Matrix 
    :param      thresh  : Stores Thresholds 
    :param      external: Input from external Neurons 
    :param      fireCount    : records how often a neuron switches to active state 

    :return             : for troubleshooting returns value before Heaviside function
    """
    sum = jCon[iter].dot(nval)
    decide = sum + external[iter] - thresh[iter]
    if sum + external[iter] - thresh[iter] > 0:
        if nval[iter] == 0:
            fireCount[iter] += 1
        nval[iter] = 1
    else:
        nval[iter] = 0
    return sum + external[iter] - thresh[iter]

def timestepMatRecord(iter, nval, jCon, thresh, external, fireCount,sizeM):
    """
    Current Calculator for whether one neuron changes value or not(31/10)

    Particularity: Records additional information ie positive and negative input 
    Sums all the input with corresponding weights. 
    Afterwords adds external input and subtracts threshold. 
    Result is plugged in Heaviside function

    :param      iter    : iterator, determines which neuron is to be changed
    :param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
    :param      jCon    : Connection Matrix 
    :param      thresh  : Stores Thresholds 
    :param      external: Input from external Neurons 
    :param      fireCount    : records how often a neuron switches to active state 

    :return         : returns positive input from other neurons, all input (-threshold) and negative input
    """
    pos = jCon[iter,:sizeM[0]].dot(nval[:sizeM[0]])
    neg = jCon[iter,sizeM[0]:].dot(nval[sizeM[0]:])
    summe = pos + neg
    sum0 = jCon[iter].dot(nval)
    if abs(sum0-summe) > 0.001:
        print("Ungenauigkeit durch Messugn")
    decide =summe + external[iter] - thresh[iter]
    if summe + external[iter] - thresh[iter] > 0:
        if nval[iter] == 0:
            fireCount[iter] += 1
        nval[iter] = 1
    else:
        nval[iter] = 0
    return [pos,summe + external[iter],neg]

def sequential(nval, jCon,  thresh, external, fireCount, indiNeuronsDetailed, recNum, sizeM):
    """
    Updates neurons in the sequence 1 to N

    if smaller than recNum (or closer to sizeMax for inhibitory) then all changes are recorded in indiNeuronsDetailed
    :param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
    :param      jCon    : Connection Matrix 
    :param      thresh  : Stores Thresholds 
    :param      external: Input from external Neurons 
    :param      fireCount    : records how often a neuron switches to active state CHANGES
    :param      time    : Controls runtime
    :param      sizeM   : Contains information over the network size
    :param      indiNeuronsDetailed:  CHANGES
    :param      recNum  : How many neurons are recorded 

    :return    Nothing 
    """
    excite = 1
    sizeMax = sum(sizeM)
    for iter in range(len(nval)):
        if excite:
            if iter <recNum:
                #indiNeuronsDetailed[iter].append(timestepMatRecord(
                vals = timestepMatRecord(
                    iter, nval, jCon, thresh, external, fireCount, sizeM)#)
                indiNeuronsDetailed[iter].append(vals)
            else:
                timestepMat (iter, nval, jCon, thresh, external, fireCount)
        else:
            if iter > sizeMax -recNum:
                indiNeuronsDetailed[sizeMax - iter-1].append(timestepMatRecord(
                    iter, nval, jCon, thresh, external, fireCount, sizeM))
            else:
                timestepMat (iter, nval, jCon, thresh, external, fireCount)

def sequRun(jCon, thresh, external, timer, sizeM, extM, K, mean0, recNum = 10):
    """
    Runs the sequence 1 to N for "time" times. 

    Manages Recordings aswell (see :return)

    :param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
    :param      jCon    : Connection Matrix 
    :param      thresh  : Stores Thresholds 
    :param      external: Input from external Neurons 
    :param      fireCount    : records how often a neuron switches to active state 
    :param      time    : Controls runtime
    :param      sizeM   : Contains information over the network size
    :param      recNum  : How many neurons are recorded 


    :return     Returns indiNeuronsDetailed and total_times_one which analyze individual (ie subset of) neurons and all neurons respectively,
                aswell as nval_over_time which records nval after every time step
    """
    debug = 1
    nval = createNval(sizeM, extM, K, mean0)  
    fireCount = np.zeros(np.shape(nval)) #Fire rate for indivual neuron
    total_times_one = np.zeros_like(nval)
    nval_over_time = [total_times_one.copy() for x in range(timer)]
    indiNeuronsDetailed = [[] for i in range(recNum)] 
    timediff = []
    for t in range(0,timer):
        timestart = time.time()
        sequential(nval, jCon, thresh, external, fireCount, indiNeuronsDetailed, recNum, sizeM)
        if timer>50:
            if t%50==0:
                print(t)
        total_times_one += (nval)
        nval_over_time[t] += nval
        timeend = time.time()
        timediff.append(timeend - timestart)
    print("mean time for one cycle")
    utils.timeOut(np.mean(np.array(timediff)))
    return indiNeuronsDetailed, total_times_one, fireCount, nval_over_time

def poissoni(sizeM, maxTime, tau, nval, jCon, thresh, external, fireCount, indiNeuronsDetailed, randomProcess, recNum):
    
    """
    Randomly chooses between excitatory or inhibitory sequence

    Randomly chooses between excitatory or inhibitory sequence with relative likelihood tau 
    to choose inhibitory (ie 1 meaning equally likely).
    Each round a new permutation of range is drawn
    Currently only supports recording individual excitatory neurons for indiNeuronsDetailed

    !Should tau be larger than one, double counting could happen

    :param      maxTime : Controls runtime
    :param      sizeM   : Contains information over the network size
    :param      tau     : How often inhibitory neurons fireCount compared to excitatory
    :param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
    :param      jCon    : Connection Matrix 
    :param      thresh  : Stores Thresholds 
    :param      external: Input from external Neurons 
    :param      fireCount    : records how often a neuron switches to active state 
    :param      indiNeuronsDetailed: 
    :param      recNum  : How many neurons are recorded 

    :return     nvalOvertime
    """
    total_times_one = np.zeros_like(nval)
    nval_over_time = np.zeros((sum(sizeM),maxTime))
    sizeMax = sum(sizeM)    
    prob =  1. / (1+tau)

    if tau > 1:
        print("Warning: tau larger than one could result in recording two events at once")

    exTime      = 0
    exIterStart = 0
    exIterMax   = sizeM[0] 
    exIter      = exIterStart 

    inTime      = 0
    inIterStart = 0
    inIterMax   = sizeM[1] 
    inIter      = inIterStart       #iterates through sequence below
    if randomProcess: 
        exSequence  = np.random.randint(0, sizeM[0], sizeM[0])
        inSequence  = np.random.randint(sizeM[0],sizeMax,sizeM[1])
    else:
        exSequence  = np.random.permutation(sizeM[0])
        inSequence  = np.random.permutation(np.arange(sizeM[0],sizeMax))

    while exTime < maxTime:
        nval_old = nval.copy()
        if np.random.uniform(0,1)<prob:
            iterator = exSequence[exIter]
            if iterator <recNum:
                vals = timestepMatRecord(iterator, nval, jCon,
                    thresh, external, fireCount, sizeM)
                indiNeuronsDetailed[iterator].append(vals)
                if vals[1] >= 1:
                    nval_over_time[iterator,exTime] += 1
            else:
                overThresh = timestepMat (iterator, nval, jCon, thresh, external, fireCount)
                if overThresh >= 0:
                    nval_over_time[iterator,exTime] += 1
            exIter +=1
            if exIter >= exIterMax:
                exTime+=1
                exIter = exIterStart
                total_times_one = nval[:sizeM[0]]
                if randomProcess: 
                    exSequence  = np.random.randint(0, sizeM[0], sizeM[0])
                else:
                    exSequence = np.random.permutation(sizeM[0])
                if exTime % 10 == 0:
                    print(exTime)
        else:
            iterator = inSequence[inIter]
            overThresh = timestepMat (iterator, nval, jCon, thresh, external, fireCount)
            if overThresh >= 0:
                nval_over_time[iterator,inTime] += 1
            inIter += 1
            if inIter >= inIterMax:
                inTime  += 1
                inIter  = inIterStart
                total_times_one   = nval[sizeM[0]:]
                if randomProcess:
                    inSequence  = np.random.randint(sizeM[0],sizeMax,sizeM[1])
                else:
                    inSequence = np.random.permutation(np.arange(sizeM[0],sizeMax))
    return nval_over_time


def poissRun(jCon, thresh, external, maxTime,
    sizeM, extM, K, mean0, tau, randomProcess, recNum = 15):
    """
    Runs a mix between poisson and sequential Code

    Introduces analyze tools


    :param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
    :param      jCon    : Connection Matrix 
    :param      thresh  : Stores Thresholds 
    :param      external: Input from external Neurons 
    :param      fireCount    : records how often a neuron switches to active state 
    :param      maxTime : Controls runtime
    :param      sizeM   : Contains information over the network size
    :param      tau     : How often inhibitory neurons fireCount compared to excitatory
    :param      recNum  : How many neurons are recorded 

    :return     Returns indiNeuronsDetailed and total_times_one which analyze individual (ie subset of) neurons and all neurons respectively

    """

    nval = createNval(sizeM, extM, K, mean0)  
    np.savetxt(f'nval{mean0*100}.csv',nval,delimiter=',')
    fireCount = np.zeros(np.shape(nval)) #Fire rate for indivual neuron
    total_times_one = np.zeros_like(nval)
    indiNeuronsDetailed = [[] for i in range(recNum)] 

    nval_over_time = poissoni(sizeM,maxTime, tau, nval, jCon, thresh, external, fireCount, indiNeuronsDetailed, randomProcess, recNum)
    total_times_one = fireCount
    return indiNeuronsDetailed, total_times_one, fireCount, nval_over_time
    



###############################################################################
########################### Setup Functions ###################################
###############################################################################
def prepare(K, mean0, tau, sizeE, sizeI, extE, extI, jE, jI, threshE, threshI):
    """
    creates all the needed objects and calls the workload functions and plots.
    Virtually a main function, without parameter definition
    """

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
    thresh = createThresh(sizeM, threshM)  
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
    utils.timeOut(timeend - timestart)
    return sizeM, threshM, extM, jCon, thresh, external


def testRoutine(
    timer, K, mean0, tau,
    sizeM,threshM, extM, recNum,
    jCon, thresh, external, figfolder, valueFolder,
    jE, jI, doSequ = 1, doPoiss = 0, doRand = 0,
    pTot=0, pIndi=0, pIndiExt= 0, pInterspike=0,  pDots=0):
    """
    executes the differen sequences
    """ 
    sizeMax = sizeM[0] + sizeM[1]
    np.set_printoptions(edgeitems = 10)
    titletxt = f'_S_{str(sizeMax)[:1]}e{int(np.log10(sizeMax))}_K_{(K)}_m0_{str(mean0)[2:]}'# \njE: {jE}   jI: {jI} '
    captiontxt = f'Network Size: {sizeMax}  K: {K}  mean_0: {mean0} \n\
        time: {timer}   jE: {jE}   jI: {jI} ' 

    print("run")
    timestart = time.time()
    extracap = ""
    extratitle = ""
    if doSequ:
        indiNeuronsDetailed, total_times_one, fireCount, nval_over_time = sequRun(
            jCon, thresh, external, timer ,sizeM, extM, K, mean0, recNum)
        extracap    = "sequence 1 to N"
        extratitle  = "sequ"
    if doPoiss or doRand:
        randomProcess = doRand
        indiNeuronsDetailed,total_times_one, fireCount, nval_over_time= poissRun(
            jCon, thresh, external, timer, sizeM, extM,
            K, mean0, tau, randomProcess, recNum)
        extracap    = "Poisson"
        extratitle  = "poiss"
        if randomProcess:
            extracap += ", stochastic"
            extratitle += "Rand"
        else:
            extracap += ", deterministic"
            extratitle += "Permute"

    captiontxt += " " + extracap 
    titletxt += "_" + extratitle if extratitle != "" else ""
    timeend = time.time()
    print("runtime of routine")
    utils.timeOut(timeend - timestart)
    
    #print(indiNeuronsDetailed)
    print("mean")
    print(np.mean(np.array(total_times_one)))
    listing = []
    for row in nval_over_time:
        listing.append(sum(row))
    print("listing")
    print(np.mean(listing))

    saveResults(valueFolder, indiNeuronsDetailed, fireCount, nval_over_time, 
                timer, threshM, titletxt, captiontxt)
    
    indiNeuronsDetailed, fireCount, nval_over_time, infoDict = recoverResults(valueFolder)
    
    meanOT = np.mean(total_times_one[:int(sizeMax/2)])/timer
    print("meanOT")
    print(meanOT)
    print("is this the same?")
    print(np.mean(nval_over_time[:sizeM[0],-5:]))
    print(np.mean(nval_over_time[sizeM[0]:,-5:]))
    ### Plotting Routine ###
    if pTot:
        plots.mean_distri(figfolder, total_times_one,fireCount, timer, titletxt, captiontxt)
    if pIndi:
        plots.indi(figfolder,indiNeuronsDetailed,fireCount, threshM, titletxt, captiontxt)
    if pIndiExt:
        plots.indiExtended(figfolder,indiNeuronsDetailed,fireCount, threshM, recNum, titletxt, captiontxt)
    if pInterspike:
        plots.interspike(figfolder, nval_over_time, timer, titletxt, captiontxt)
    if pDots:
        plots.dots(figfolder, nval_over_time, timer, titletxt, captiontxt)
    
    del indiNeuronsDetailed
    del fireCount
    del nval_over_time
    del total_times_one   
    return meanOT


def afterSimulationAnalysis():
    useMostRecent = 1
    vfolder = Path("ValueVault")
    if useMostRecent:
        loadTimeStr = utils.mostRecent(vfolder)
    else:
        loadTimeStr = "testreihe_191111_1038" #create a "find most recent" function
    valueFolder =  vfolder / loadTimeStr

    timestr = time.strftime("%y%m%d_%H%M")
    figfolder = "figs/testreihe_" + timestr

    showRange = 15
    indiNeuronsDetailed, fireCount, nval_over_time, infoDict = recoverResults(valueFolder)
    total_times_one = utils.rowSums((nval_over_time))
    print(np.mean(total_times_one))
    print(np.shape(indiNeuronsDetailed))


    titletxt    = infoDict["titletxt"]
    captiontxt  = infoDict["captiontxt"]
    threshM     = infoDict["threshM"]
    timer       = infoDict["timer"]

    ### Plotting Routine ###
    pTot        = 0
    pIndi       = 0
    pIndiExt    = 0
    pInterspike = 0
    pDots       = 1

    if pTot:
        plots.mean_distri(figfolder, total_times_one,fireCount, timer, titletxt, captiontxt)
    if pIndi:
        plots.indi(figfolder,indiNeuronsDetailed,fireCount, threshM, titletxt, captiontxt)
    if pIndiExt:
        plots.indiExtended(figfolder,indiNeuronsDetailed,fireCount, threshM, recNum, titletxt, captiontxt)
    if pInterspike:
        plots.interspike(figfolder, nval_over_time, timer, titletxt, captiontxt)
    if pDots:
        plots.dots(figfolder, nval_over_time, timer, titletxt, captiontxt)



def simParam():
    timestr = time.strftime("%y%m%d_%H%M")
    figfolder = "figs/testreihe_" + timestr
    valuefoldername = "ValueVault/testreihe_"
    valueFolder =  Path(valuefoldername + timestr)
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
    #Recording Specification(should be reviewed):
    recNum      = 100
    ### Deviations ###
    timer   = 30
    K       = 1000
    size    = 1000
    sizeE   = size
    sizeI   = size
    return timer, sizeE, sizeI, extE, extI, jE, jI, threshE, threshI, tau, mean0, K, recNum, figfolder, valueFolder 
    
def main():
    timer, sizeE, sizeI, extE, extI, jE, jI, threshE, threshI,\
         tau, mean0, K, figfolder, valueFolder \
             = simParam()
    #Bools for if should be plotted or not
    pTot        = 1
    pIndiExt    = 1
    pInterspike = 1
    pDots       = 1

    #Only one Sequence per Routine
    doSequ      = 0
    doPoiss     = 1 
    doRand      = 0 

    sizeM, threshM, extM, jCon, thresh, external \
        = prepare(  K, mean0, tau, sizeE, sizeI, extE, extI,
                    jE, jI,threshE, threshI)
    sizeMax = np.sum(sizeM)
    testRoutine(
        timer, K, mean0, tau,
        sizeM, threshM, extM, recNum,
        jCon, thresh, external, figfolder, valueFolder,
        jE, jI, doSequ, doPoiss, doRand,
        pTot,  pIndiExt, pInterspike,pDots)
if __name__ == "__main__":
    #afterSimulationAnalysis()
    main()
    
    # mean100 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
    #         0.09, 0.1  , 0.125, 0.15 , 0.175, 0.2  , 0.225, 0.25 , 0.275] 
    # mean100 = [0.01, 0.04]# 0.1, 0.2]

    # meanOT = []
    # for i,mean0_itt in enumerate(mean100):
        # meanOT.append(
        # print(meanOT)
    # mean100+=meanOT
    # mean100 = np.array(mean100)
    # mean100.shape = (2,len(meanOT))
    # print(mean100)
    # np.save("mean0vsmeanOT",mean100)