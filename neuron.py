"""
This is the VanillaMain python file for my project

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
import old


################################################################################
############################## Global Variables ################################
################################################################################
load_jConMatrix_from_can_GLOBAL = 0
plotLoc_print2console_GLOBAL = 1
################################################################################
############################## Utility Functions ###############################
################################################################################

def saveResults(valueFolder, indiNeuronsDetailed, fireCount, nval_over_time, 
                timer, threshM, shorttxt, captiontxt):
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
        "shorttxt"  : shorttxt,
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
        raise ValueError("proboverThreshability assumes equal likelihood of being excitatory or inhibitory")

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


def createNval(sizeM, activeAtStart):
    """
    Initializes neuron values "nval" with starting values

    :param      sizeM   : Contains size of exhib and inhib neurons
    :param      extM    : Contains factors of external neurons for the inhib values in the system
    :param      K       : Connection Number
    :param      meanExt   : Mean activation of external neurons
    """
    nval = []
    ones = activeAtStart  * sizeM
    for i in range(len(sizeM)):
        numof1 = int(ones[i])
        numof0 = sizeM[i] - numof1
        arr = [0] * numof0 + [1] * numof1
        arr = random.sample(arr,len(arr))
        nval+= arr
    return np.array(nval)

def createConstantThresh(sizeM, threshM):
    """
    Creates Threshold vector with threshold for each Datapoint

    :param      sizeM   : Contains size of exhib and inhib neurons
    :param      threshM : Contains values for threshold
    """
    thresh= []
    for i in range(2):
        thresh.extend([threshM[i] for x in range(sizeM[i])])
    return np.array(thresh)

def createGaussThresh(sizeM,threshM):
    dev = 0.3
    thresh = []
    for i in range(len(sizeM)):
        thresh +=  [np.random.normal(threshM[i],dev) for x in range(sizeM[i])]
    return np.array(thresh)

def createBoundThresh(sizeM,threshM):
    delta = 0.3
    thresh = []
    for i in range(len(sizeM)):
        thresh +=  [np.random.uniform(threshM[i]-delta/2,threshM[i]+delta/2) for x in range(sizeM[i])]
    return np.array(thresh)

def createExt(sizeM, extM, K, meanExt):
    """
    Creates vector of external input for each Datapoint 
    
    (with all exhib and all inhib having the same value)

    :param      sizeM   : Contains size of exhib and inhib neurons
    :param      extM    : Contains factors of external neurons for the inhib values in the system
    :param      K       : Connection Number
    :param      meanExt   : Mean activation of external neurons
    """
    ext = []
    extVal = extM * math.sqrt(K) *meanExt
    for i in range(len(sizeM)):
        ext.extend([extVal[i] for x in range(sizeM[i])])
    return np.array(ext)


################################################################################
############################## Core Functions ##################################
################################################################################
def timestepMat (iter, nval, jCon, thresh, external, fireCount, recordPrecisely=0,combMinSize=[0],combMaxSize=[0]):
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
    if decide > 0:
        if nval[iter] == 0:
            fireCount[iter] += 1
        nval[iter] = 1
    else:
        nval[iter] = 0
    if recordPrecisely:
        inputs = []
        for i in range(len(combMinSize)):
            inputs.append(  jCon[iter,combMinSize[i]:combMaxSize[i]].\
                        dot(nval[combMinSize[i]:combMaxSize[i]]))
        return [inputs[0], (decide + thresh[iter]), inputs[1]]
    return decide 


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
    nval_over_time = np.zeros((sum(sizeM),int(maxTime/tau*1.1)))
    sizeMax = sum(sizeM)    
    likelihood_of_choosing_excite =  tau / (1+tau)
    ### New record containers ###
    activeOT = [[] for _ in range(sizeMax)]
    fireOT   = [[] for _ in range(sizeMax)]
    justActive=[0 for _ in range(sizeMax)]

    comb_Big_Time   = [0, 0]
    comb_Small_Time = [0, 0]       #iterates through sequence below
    combMinSize     = np.array([0, sizeM[0]])
    combMaxSize     = combMinSize + sizeM
    combRange       = [np.arange(combMinSize[i],combMaxSize[i]) for i in range(2)]
    combSequence    = []
    for inhibite in range(2):
        if randomProcess: 
            combSequence.append( np.random.randint(combMinSize[inhibite],combMaxSize[inhibite], sizeM[inhibite]))
        else:
            combSequence.append( np.random.permutation(combRange[inhibite]))

    while comb_Big_Time[0] < maxTime:
        if np.random.uniform(0,1)<likelihood_of_choosing_excite:
            inhibite = 0
        else: 
            inhibite = 1
        iterator = combSequence[inhibite][comb_Small_Time[inhibite]]
        if iterator <recNum:
            recordPrecisely = True
        else: 
            recordPrecisely = False
        vals = timestepMat(iterator, nval, jCon,
                thresh, external, fireCount, recordPrecisely,
                combMinSize, combMaxSize)
        if isinstance(vals, list):
            indiNeuronsDetailed[iterator].append(vals)
            vals = vals[1] - thresh[iterator]
        if vals >= 0:
            nval_over_time[iterator,comb_Big_Time[inhibite]] += 1
            temp = comb_Big_Time[0]+comb_Small_Time[0]/sizeM[0]
            activeOT[iterator].append(temp)
            if not justActive[iterator]:
                fireOT[iterator].append(temp)
            justActive[iterator] = 1
        else:
            justActive[iterator] = 0

        comb_Small_Time[inhibite] +=1
        if comb_Small_Time[inhibite] >= sizeM[inhibite]:
            comb_Big_Time[inhibite] +=1
            comb_Small_Time[inhibite] = 0
            if randomProcess: 
                combSequence[inhibite]  = np.random.randint(combMinSize[inhibite],combMaxSize[inhibite], sizeM[inhibite])
            else:
                combSequence[inhibite] = np.random.permutation(combRange[inhibite])
            if comb_Big_Time[0] % 10 == 0 and not inhibite:
                print(f"{(comb_Big_Time[0]/maxTime):.0%}", end=", ")
    print("")
    return nval_over_time, activeOT, fireOT


def poissRun(jCon, thresh, external,  sizeM, extM,
    maxTime, K, meanActi_at_0, meanExt, tau, randomProcess, recNum = 15):
    """
    #Obsolete Wrapper
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

    nval = createNval(sizeM, meanActi_at_0)  
    nval0 = nval.copy()
    #np.savetxt(f'nval{meanExt*100}.csv',nval,delimiter=',')
    fireCount = np.zeros(np.shape(nval)) #Fire rate for indivual neuron
    total_times_one = np.zeros_like(nval)
    indiNeuronsDetailed = [[] for i in range(recNum)] 

    nval_over_time, activeOT, fireOT = poissoni(sizeM,maxTime, tau, nval, jCon, thresh, external, fireCount, indiNeuronsDetailed, randomProcess, recNum)
    total_times_one = fireCount
    return indiNeuronsDetailed, total_times_one, fireCount, nval_over_time, activeOT, fireOT, nval0
    



###############################################################################
########################### Setup Functions ###################################
###############################################################################
def prepare(K, meanExt, tau, sizeE, sizeI, extE, extI, jE, jI, threshE, threshI, doThresh):
    """
    creates all the needed objects and calls the workload functions and plots.
    Virtually a VanillaMain function, without parameter definition

    """
    sizeM = np.array([sizeE, sizeI])
    sizeM.setflags(write=False)
    sizeMax = sizeE + sizeI

    if min(sizeM)<K:
        raise ValueError("K must be smaller than or equal to size "
                        +"of excitatory or inhibitory Values")
    ### External Input ###
    extM = np.array([extE, extI])
    extM.setflags(write=False)
    external = createExt(sizeM,extM, K, meanExt)     
    external.setflags(write=False)

    ### Threshoold Level ###
    threshM = np.array([threshE, threshI])
    threshM.setflags(write=False)
    if   (doThresh == "constant"): thresh = createConstantThresh(sizeM, threshM)  
    elif (doThresh == "gauss"   ): thresh = createGaussThresh(sizeM, threshM)  
    elif (doThresh == "bound"   ): thresh = createBoundThresh(sizeM, threshM)  
    else: raise NameError ("Invalid threshold codeword selected")
    thresh.setflags(write=False)

    ### Values of Connection Matrix ###
    jVal = np.array([[1, -1*jE],[1, -1*jI]])
    jVal = jVal/math.sqrt(K)

    ### Connection Matrix ###
    print("Create jCon")
    timestart = time.time()
    jCon = createjCon(sizeM, jVal,K)
    jCon.setflags(write=False)
    timeend = time.time()
    utils.timeOut(timeend - timestart)
    return sizeM, threshM, extM, jCon, thresh, external


def run_container(
    timer, K, meanExt, tau, meanActi_at_0,
    sizeM,threshM, extM, recNum,
    jCon, thresh, external, figfolder, valueFolder,
    jE, jI, doPoiss = 0, doRand = 0, doThresh = "constant",
    nDistri=0, nMeanOT=0, nInter=0,
    pActiDist=0, pIndiExt= 0, pInterspike=0,  pDots=0,pMeanOT=0):
    """
    executes the differen sequences
    """ 
    sizeMax = sizeM[0] + sizeM[1]
    np.set_printoptions(edgeitems = 10)
    captiontxt = f'Network Size: {sizeMax}  K: {K}  mean_0: {meanExt} \n\
        time: {timer}   jE: {jE}   jI: {jI}' 
    shorttxt   = f'_S{int(np.log10(sizeMax))}'\
                + f'_K{int(np.log10(K))}_m{str(meanExt)[2:]}_t{str(timer)[:-1]}' # \njE: {jE}   jI: {jI} ' 


    print("run")
    timestart = time.time()

    if doPoiss or doRand:
        randomProcess = doRand
    ### Calling the Actual function ###
        indiNeuronsDetailed,total_times_one, fireCount, nval_over_time, activeOT, fireOT, nval0= poissRun(
            jCon, thresh, external, sizeM, extM, timer,
            K, meanActi_at_0, meanExt, tau, randomProcess, recNum)
    ### Change name of caption and filename according to specification ###
        if randomProcess:
            captiontxt += f",\n stochastic Updates"
            shorttxt += "_rY"
        else:
            captiontxt += ",\n deterministic Updates"
            shorttxt += "_rN"
    else:
        raise ValueError("Neither doPoiss nor doRand == True: nothing to do")
         
    ### still updating caption and title ###
    if   (doThresh == "constant"): 
        captiontxt += ", Thresholds = constant"
        shorttxt += "_tC"
    elif (doThresh == "gauss"   ):   
        captiontxt += ", Thresholds = gaussian"
        shorttxt += "_tG"
    elif (doThresh == "bound"   ):  
        captiontxt += ", Thresholds = bounded"
        shorttxt += "_tB"

    ### still updating caption and title ###
    figfolder += shorttxt 
    valueFolder = Path(str(valueFolder) + shorttxt)
    plots.figfolder_GLOBAL  = figfolder
    plots.captiontxt_GLOBAL = captiontxt
    plots.titletxt_GLOBAL   = shorttxt

    ### time check ###
    timeend = time.time()
    print("runtime of routine")
    utils.timeOut(timeend - timestart)

    saveResults(valueFolder, indiNeuronsDetailed, fireCount, nval_over_time, 
                timer, threshM, shorttxt, captiontxt)
    
    ### Plotting Routine ###
    if pActiDist:
        plots.activation_distribution(figfolder, total_times_one,fireCount, timer, shorttxt, captiontxt)
    if pIndiExt:
        plots.indiExtended(figfolder,indiNeuronsDetailed,fireCount, threshM, recNum, shorttxt, captiontxt)
    if pInterspike:
        plots.interspike(figfolder, nval_over_time, timer, shorttxt, captiontxt)
    if pDots:
        plots.dots(figfolder, nval_over_time, timer, shorttxt, captiontxt)
    if pMeanOT:
        plots.meanOT(figfolder, nval_over_time, sizeM, timer, shorttxt, captiontxt)
    if nDistri:
        plots.newDistri(activeOT, timer)
    if nMeanOT:
        plots.newMeanOT(activeOT,sizeM)
    if nInter:
        plots.newInterspike(fireOT,timer)

    # lenofactive = np.sum([len(row) for row in activeOT])
    # lenoffire = np.sum([len(row) for row in fireOT])
    meanActivationOT = np.mean(nval_over_time[:sizeM[0],-1])
    return meanActivationOT
def FunMain():
    (timer,  sizeE, sizeI, extE, extI, jE, jI, threshE, threshI,
        meanActi_at_0,tau, meanExt, K, recNum, figfolder, valueFolder
    ) = numParam()

    (pActiDist , pIndiExt, pInterspike, pDots, pMeanOT, nDistri, nMeanOT, nInter,
        doPoiss, doRand, doThresh
    ) = doParam()
    (sizeM, threshM, extM, jCon, thresh, external
    ) = prepare(K, meanExt, tau, sizeE, sizeI, extE, extI,
                jE, jI,threshE, threshI, doThresh)
    mean100 = [0.01, 0.04, 0.1, 0.2]
    meanOT = []
    for i,mean0_itt in enumerate(mean100):
        meanOT.append(
    run_container(
                timer, K, meanExt, tau, meanActi_at_0,
                sizeM, threshM, extM, recNum,
                jCon, thresh, external, figfolder, valueFolder,
                jE, jI, doPoiss, doRand, doThresh,
                nDistri, nMeanOT, nInter,
                pActiDist,  pIndiExt, pInterspike,pDots, pMeanOT)
        )
    mean100+=meanOT
    mean100 = np.array(mean100)
    mean100.shape = (2,len(meanOT))
    print(mean100)

def VanillaMain():
    (timer,  sizeE, sizeI, extE, extI, jE, jI, threshE, threshI,
        meanActi_at_0,tau, meanExt, K, recNum, figfolder, valueFolder
    ) = numParam()
    (pActiDist , pIndiExt, pInterspike, pDots, pMeanOT, nDistri, nMeanOT, nInter,
        doPoiss, doRand, doThresh
    ) = doParam()
    (sizeM, threshM, extM, jCon, thresh, external
    ) = prepare(K, meanExt, tau, sizeE, sizeI, extE, extI,
                jE, jI,threshE, threshI, doThresh)
    run_container(
        timer, K, meanExt, tau, meanActi_at_0,
        sizeM, threshM, extM, recNum,
        jCon, thresh, external, figfolder, valueFolder,
        jE, jI, doPoiss, doRand, doThresh,
        nDistri, nMeanOT, nInter,
        pActiDist,  pIndiExt, pInterspike,pDots, pMeanOT)

def numParam():
    """
    Sets all parameters relevant to the simulation    

    For historic reasons also sets the folder where figures and data are saved
    """
    timestr = time.strftime("%y%m%d_%H%M")
    figfolder = "figs/testreihe_" + timestr
    valuefoldername = "ValueVault/testreihe_"
    valueFolder     =  Path(valuefoldername + timestr)
    timer           = 200
    sizeE           = 10000
    sizeI           = 10000
    extE            = 1.
    extI            = 0.8
    jE              = 2.
    jI              = 1.8
    threshE         = 1.
    threshI         = 0.7
    tau             = 0.9
    meanExt         = 0.1
    K               = 1000
    meanActi_at_0   = 0.1
    recNum          = 20
    ### Most changed vars ###
    timer           = 60
    K               = 1000
    size            = 2000
    sizeE,sizeI     = size,size

    return  timer, sizeE, sizeI, extE, extI, jE, jI, threshE, threshI,\
            meanActi_at_0, tau, meanExt, K, recNum, figfolder, valueFolder 
def doParam():
    """
    specifies most behaviors of 
    """
    #Bools for if should be plotted or not
    #pActiDist   = 0 # unexplained discrepancies
    pIndiExt    = 1
    pInterspike = 1 # obsolete
    pDots       = 1
    pMeanOT     = 1 # obsolete
    nDistri     = 1
    nMeanOT     = 1
    nInter      = 1

    doThresh    = "constant" #"constant", "gauss", "bound"

    #Only one Sequence per Routine
    doPoiss     = 1 
    doRand      = 0 

    plots.savefig_GLOBAL    = 1
    plots.showPlots_GLOBAL  = 0

    return (pActiDist , pIndiExt, pInterspike, pDots, pMeanOT, nDistri, nMeanOT, nInter,
            doPoiss, doRand, doThresh)

if __name__ == "__main__":
    #FunMain()
    VanillaMain()

###############################################################################
################################## To Dos #####################################
###############################################################################
"""
Rewrite Recover and SaveResults\
Remove Obsolete Plots
Plot meanOT vs meanExt


Cython someday
"""