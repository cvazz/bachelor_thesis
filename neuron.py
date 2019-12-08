###############################################################################
############################## Imported Modules ###############################
###############################################################################
### Numbers ###
import numpy as np
import math
import random 
import scipy.stats as st

import matplotlib.pyplot as plt

### File Interaction and Manipulation ###
from pathlib import Path
import pickle

### Duh ###
import time

### Local Modules ###
import mathsim as msim
import utils
import plots
import old

###############################################################################
############################## Utility Functions ##############################
###############################################################################

def saveResults(valueFolder, indiNeuronsDetailed, 
                activeOT, fireOT, info, toDo):
    """
    Save recordings of output to file.
    
    :param valueFolder: Path to storage folder
    :type valueFolder: class: 'pathlib.PosixPath'
    :param indiNeuronsDetailed:  

    """
    if not valueFolder.exists():
        valueFolder.mkdir(parents = True)


    remember_list = [info, toDo]
    indiNametxt     = "indiNeurons"
    infoNametxt     = "infoDict"
    activeNametxt   = "activeOT"    
    fireNametxt     = "fireOT"      

    indiName        = utils.makeNewPath(valueFolder, indiNametxt, "npy")
    fireName        = utils.makeNewPath(valueFolder, fireNametxt, "npy")
    activeName      = utils.makeNewPath(valueFolder, activeNametxt, "npy")
    infoName        = utils.makeNewPath(valueFolder, infoNametxt, "pkl")

    np.save(indiName, indiNeuronsDetailed)
    np.save(fireName, fireOT)
    np.save(activeName, activeOT)
    infoName.touch()
    with open(infoName, "wb") as infoFile:
        pickle.dump(remember_list, infoFile, protocol = pickle.HIGHEST_PROTOCOL)

def recoverResults(valueFolder):
    """
    Load saved results from file.

    Missing functionality: so far, does only take neurons with preset names

    :param valueFolder: |valueFolder_desc|
    :return: indiNeuronsDetailed,  
    """

    indiNametxt     = "indiNeurons"
    infoNametxt     = "infoDict"
    activeNametxt   = "activeOT"    
    fireNametxt     = "fireOT"      

    indiName        = utils.makeExistingPath(valueFolder, indiNametxt, "npy")
    activeName      = utils.makeExistingPath(valueFolder, activeNametxt, "npy")
    fireName        = utils.makeExistingPath(valueFolder, fireNametxt, "npy")
    infoName        = utils.makeExistingPath(valueFolder, infoNametxt, "pkl")

    indiNeuronsDetailed = np.load(indiName, allow_pickle = True)
    activeOT            = np.load(activeName, allow_pickle = True)
    fireOT              = np.load(fireName, allow_pickle = True)
   

    with open(infoName, 'rb') as infoFile:
        info, toDo = pickle.load(infoFile)

    return indiNeuronsDetailed, activeOT, fireOT, info, toDo

################################################################################
############################ Creating Functions ################################
################################################################################
def createjCon(sizeM, jE, jI, K):
    """
    Current Connection Matrix Creator (31/10)

    Only working for 

    :param     sizeM   : Contains size of exhib and inhib
    :param     jVal    : Contains nonzero Values for Matrix
    :param     K       : Number of connections with inhib/exhib

    :return     jCon    : Connection Matrix
    """
    if sizeM[0] != sizeM[1]:
        raise ValueError("proboverThreshability assumes equal likelihood "
                        +"of being excitatory or inhibitory")

    print("Create jCon")
    timestart = time.time()
    sizeMax     = sizeM[0] + sizeM[1]
    jVal = np.array([[1, -1*jE],[1, -1*jI]])
    jVal = jVal/math.sqrt(K)

    ### Connection Matrix ###

    oddsBeingOne= 2*K/sizeMax
    jCon        = np.random.binomial(1, oddsBeingOne, sizeMax**2)

    jCon        = jCon.astype(float)
    jCon.shape  = (sizeMax,sizeMax)

    #add weights
    jCon[:sizeM[0],:sizeM[0]] = np.multiply(jCon[:sizeM[0],:sizeM[0]],jVal[0,0])
    jCon[sizeM[0]:,:sizeM[0]] = jCon[sizeM[0]:,:sizeM[0]]*jVal[1,0]
    jCon[:sizeM[0],sizeM[0]:] = jCon[:sizeM[0],sizeM[0]:]*jVal[0,1]
    jCon[sizeM[0]:,sizeM[0]:] = jCon[sizeM[0]:,sizeM[0]:]*jVal[1,1]

    jCon.setflags(write=False)
    timeend = time.time()
    utils.timeOut(timeend - timestart)
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

def createThresh(sizeM, threshM, doThresh):
    if   (doThresh == "constant"): thresh = createConstantThresh(sizeM, threshM)  
    elif (doThresh == "gauss"   ): thresh = createGaussThresh(sizeM, threshM)  
    elif (doThresh == "bound"   ): thresh = createBoundThresh(sizeM, threshM)  
    else: raise NameError ("Invalid threshold codeword selected")
    thresh.setflags(write=False)
    return thresh

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
        thresh +=  [np.random.uniform(threshM[i]-delta/2,threshM[i]+delta/2)
                    for x in range(sizeM[i])]
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
    external = np.array(ext)
    return external

################################################################################
############################## Analysis Tools ##################################
################################################################################
def analyzeMeanOT(inputOT,sizeM):
    # Get an array of ints for excitatory and inhibitory,
    # each int represents one activation/firing at this time point
    # int is only one level of precision. It could be modified
    flat_active = [np.array([x for row in inputOT[:sizeM[0]] for x in row],dtype=int),
            np.array([x for row in inputOT[sizeM[0]:] for x in row],dtype=int)]
    # Count how many individuals have fired at a given time
    uniq_active = [np.unique(act, return_counts = 1) for act in flat_active]
    
    # Convert count above into a value between 0 and 1
    normalized_active = [uniq_active[i][1]/sizeM[i] for i in range(2)]
    
    return normalized_active

################################################################################
############################## Core Functions ##################################
################################################################################
def timestepMat (iter, nval, jCon, thresh, external,
                 recordPrecisely=0,combMinSize=[0],combMaxSize=[0]):
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

    :return             : for troubleshooting returns value before Heaviside function
    """
    sum = jCon[iter].dot(nval)
    decide = sum + external[iter] - thresh[iter]
    nval[iter] = int(decide > 0)
    # if decide > 0:
    #     nval[iter] = 1
    # else:
    #     nval[iter] = 0
    if recordPrecisely:
        inputs = []
        for i in range(len(combMinSize)):
            inputs.append(  jCon[iter,combMinSize[i]:combMaxSize[i]].\
                        dot(nval[combMinSize[i]:combMaxSize[i]]))
        return [inputs[0]+external[iter], (decide + thresh[iter]), inputs[1]]
    return decide 


def update_org_det( nval, jCon, thresh, external,
                    indiNeuronsDetailed, randomProcess, info):
    (sizeM, maxTime, tau, recNum) = info["sizeM"],info["timer"], info["tau"], info['recNum']
    sizeMax = sum(sizeM)    
    
    class my_pdf(st.rv_continuous):
        def _pdf(self,x, tau):
            return x*math.exp(-x/tau)/(tau**2)  # Normalized over its range, in this case [0,1]

    dist_R = my_pdf(a=0, name='my_pdf')
    ### New record containers ###
    activeOT = [[] for _ in range(sizeMax)]
    fireOT   = [[] for _ in range(sizeMax)]

    comb_Big_Time   = [0, 0]        # is added up maxTime times before hard stop
    comb_Small_Time = [0, 0]        # is added up N times before being resetted 
    combMinSize     = np.array([0, sizeM[0]])
    combMaxSize     = combMinSize + sizeM
    combRange       = [np.arange(combMinSize[i],combMaxSize[i]) for i in range(2)]
    combSequence    = []
    combDelta       = []
    for inhibite in range(2):
        combDelta.append( np.random.uniform(0,1,sizeM[inhibite]))
        combDelta[inhibite].sort()
        combSequence.append( np.random.permutation(combRange[inhibite]))


    while comb_Big_Time[0] < maxTime:
        # inhibite = 0 with likelihood of choosing excite
        inhibite = int((comb_Big_Time[0]+combDelta[0][comb_Small_Time[0]])*dist_R >
                      (comb_Big_Time[1]+combDelta[1][comb_Small_Time[1]])*tau)
        # chooses the next neuron to be iterated through
        iterator = (combSequence[inhibite]
        [comb_Small_Time[inhibite]])
        # records the first "recNum" values
        recordPrecisely = iterator <recNum
        # checks whether the neuron was just active
        justActive = nval[iterator] 

        result = timestepMat(iterator, nval, jCon,
                thresh, external, recordPrecisely,
                combMinSize, combMaxSize)
        ### if result is of type list it needs to be recorded ...
        if isinstance(result, list):
            indiNeuronsDetailed[iterator].append(result)
            # ... and converted back to a float value
            result = result[1] - thresh[iterator]
        ### Record Activation
        if result >= 0:
            temp = comb_Big_Time[0]+comb_Small_Time[0]/sizeM[0]
            activeOT[iterator].append(temp)
            if not justActive:
                fireOT[iterator].append(temp)

        comb_Small_Time[inhibite] +=1
        ### End of comb_Small_Time Sequence
        if comb_Small_Time[inhibite] >= sizeM[inhibite]:
            comb_Big_Time[inhibite] +=1
            comb_Small_Time[inhibite] = 0
            if comb_Big_Time[0] % 10 == 0 and not inhibite:
                print(f"{(comb_Big_Time[0]/maxTime):.0%}", end=", ", flush=True)
    print("")

    return  activeOT, fireOT

def update_org( nval, jCon, thresh, external,
                indiNeuronsDetailed, randomProcess, info):
    """
    Selects the sequence of updates and records results on the fly

    Randomly chooses between excitatory or inhibitory sequence with relative likelihood tau 
    to choose inhibitory (ie 1 meaning equally likely).
    Each round a new permutation of range is drawn
    Currently only supports recording individual excitatory neurons for indiNeuronsDetailed


    :param      maxTime : Controls runtime
    :param      sizeM   : Contains information over the network size
    :param      tau     : How often inhibitory neurons fire compared to excitatory
    :param      nval    : current values of all neurons, is CHANGED to reflect new value within function 
    :param      jCon    : Connection Matrix 
    :param      thresh  : Stores Thresholds 
    :param      external: Input from external Neurons 
    :param      indiNeuronsDetailed: 
    :param      recNum  : How many neurons are recorded 

    :return     nvalOvertime
    """



    (sizeM, maxTime, tau, recNum) = info["sizeM"],info["timer"], info["tau"], info['recNum']
    sizeMax = sum(sizeM)    
    likelihood_of_choosing_excite =  tau / (1+tau)
    ### New record containers ###
    activeOT = [[] for _ in range(sizeMax)]
    fireOT   = [[] for _ in range(sizeMax)]

    comb_Big_Time   = [0, 0]        # is added up maxTime times before hard stop
    comb_Small_Time = [0, 0]        # is added up N times before being resetted 
    combMinSize     = np.array([0, sizeM[0]])
    combMaxSize     = combMinSize + sizeM
    combRange       = [np.arange(combMinSize[i],combMaxSize[i]) for i in range(2)]
    combSequence    = []
    for inhibite in range(2):
        if randomProcess: 
            combSequence.append(np.random.randint(
                    combMinSize[inhibite],combMaxSize[inhibite], sizeM[inhibite]))
        else:
            combSequence.append( np.random.permutation(combRange[inhibite]))

    while comb_Big_Time[0] < maxTime:
        # inhibite = 0 with likelihood of choosing excite
        inhibite = int(np.random.uniform(0,1)>likelihood_of_choosing_excite)
        # chooses the next neuron to be iterated through
        iterator = combSequence[inhibite][comb_Small_Time[inhibite]]
        # records the first "recNum" values
        recordPrecisely = iterator <recNum
        # checks whether the neuron was just active
        justActive = nval[iterator] 

        result = timestepMat(iterator, nval, jCon,
                thresh, external,  recordPrecisely,
                combMinSize, combMaxSize)
        ### if result is of type list it needs to be recorded ...
        if isinstance(result, list):
            indiNeuronsDetailed[iterator].append(result)
            # ... and converted back to a float value
            result = result[1] - thresh[iterator]
        ### Record Activation
        if result >= 0:
            temp = comb_Big_Time[0]+comb_Small_Time[0]/sizeM[0]
            activeOT[iterator].append(temp)
            if not justActive:
                fireOT[iterator].append(temp)

        comb_Small_Time[inhibite] +=1
        ### End of comb_Small_Time Sequence
        if comb_Small_Time[inhibite] >= sizeM[inhibite]:
            comb_Big_Time[inhibite] +=1
            comb_Small_Time[inhibite] = 0
            if randomProcess: 
                combSequence[inhibite]  = np.random.randint(
                    combMinSize[inhibite],combMaxSize[inhibite], sizeM[inhibite])
            else:
                combSequence[inhibite] = np.random.permutation(combRange[inhibite])
            if comb_Big_Time[0] % 10 == 0 and not inhibite:
                print(f"{(comb_Big_Time[0]/maxTime):.0%}", end=", ", flush=True)
    print("")
    return  activeOT, fireOT

def update_org_for( nval, jCon, thresh, external,
                indiNeuronsDetailed, randomProcess, info):



    (sizeM, maxTime, tau, recNum) = info["sizeM"],info["timer"], info["tau"], info['recNum']
    sizeMax = sum(sizeM)    
    likelihood_of_choosing_excite =  tau / (1+tau)
    ### New record containers ###
    activeOT = [[] for _ in range(sizeMax)]
    fireOT   = [[] for _ in range(sizeMax)]

    comb_Big_Time   = [0, 0]        # is added up maxTime times before hard stop
    comb_Small_Time = [0, 0]        # is added up N times before being resetted 
    combMinSize     = np.array([0, sizeM[0]])
    combMaxSize     = combMinSize + sizeM
    combRange       = [np.arange(combMinSize[i],combMaxSize[i]) for i in range(2)]
    combSequence    = []
    for inhibite in range(2):
        if randomProcess: 
            combSequence.append(np.random.randint(
                    combMinSize[inhibite],combMaxSize[inhibite], sizeM[inhibite]))
        else:
            combSequence.append( np.random.permutation(combRange[inhibite]))

    while comb_Big_Time[0] < maxTime:
        # inhibite = 0 with likelihood of choosing excite
        inhibite = int(np.random.uniform(0,1)>likelihood_of_choosing_excite)
        # chooses the next neuron to be iterated through
        iterator = combSequence[inhibite][comb_Small_Time[inhibite]]
        # records the first "recNum" values
        recordPrecisely = iterator <recNum
        # checks whether the neuron was just active
        justActive = nval[iterator] 

        result = timestepMat(iterator, nval, jCon,
                thresh, external,  recordPrecisely,
                combMinSize, combMaxSize)
        ### if result is of type list it needs to be recorded ...
        if isinstance(result, list):
            indiNeuronsDetailed[iterator].append(result)
            # ... and converted back to a float value
            result = result[1] - thresh[iterator]
        ### Record Activation
        if result >= 0:
            temp = comb_Big_Time[0]+comb_Small_Time[0]/sizeM[0]
            activeOT[iterator].append(temp)
            if not justActive:
                fireOT[iterator].append(temp)

        comb_Small_Time[inhibite] +=1
        ### End of comb_Small_Time Sequence
        if comb_Small_Time[inhibite] >= sizeM[inhibite]:
            comb_Big_Time[inhibite] +=1
            comb_Small_Time[inhibite] = 0
            if randomProcess: 
                combSequence[inhibite]  = np.random.randint(
                    combMinSize[inhibite],combMaxSize[inhibite], sizeM[inhibite])
            else:
                combSequence[inhibite] = np.random.permutation(combRange[inhibite])
            if comb_Big_Time[0] % 10 == 0 and not inhibite:
                print(f"{(comb_Big_Time[0]/maxTime):.0%}", end=", ", flush=True)
    print("")
    return  activeOT, fireOT

###############################################################################
########################### Setup Functions ###################################
###############################################################################
def prepare(info, toDo):
    """
    creates all the needed objects and calls the workload functions and plots.
    Virtually a VanillaMain function, without parameter definition

    """
    if min(info['sizeM'])<info["K"]:
        raise Warning("K must be smaller than or equal to size "
                        +"of excitatory or inhibitory Values")
    external = createExt(info["sizeM"],info["extM"], info["K"], info["meanExt"])     
    thresh = createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
    jCon = createjCon(info["sizeM"], info["jE"], info["jI"], info["K"])

    return jCon, thresh, external


def run_box( jCon, thresh, external, info, toDo ):
    """
    executes the differen sequences
    """ 
    print("run")
    timestart = time.time()
    valueFolder = describe( toDo, info, 0 )

    ### Th function ###
    nval = createNval(info["sizeM"], info["meanStartActi"])  
    nval0 = nval.copy()
    indiNeuronsDetailed = [[] for i in range(info['recNum'])] 
    if toDo["doDet"]:
        activeOT, fireOT = update_org_det( nval, jCon, thresh, external,
                                  indiNeuronsDetailed, toDo["doRand"], info)
    else:
        activeOT, fireOT = update_org( nval, jCon, thresh, external,
                                  indiNeuronsDetailed, toDo["doRand"], info)

    ### time check ###
    timeend = time.time()
    print("runtime of routine")
    utils.timeOut(timeend - timestart)

    saveResults(valueFolder, indiNeuronsDetailed, 
                activeOT, fireOT, info, toDo)
    # (indiNeuronsDetailed, activeOT, fireOT, nval_OT, info, toDo
    # )= recoverResults(valueFolder)

    return (indiNeuronsDetailed,   
            activeOT, fireOT, nval0)

def describe( toDo, info, figs ):

    sizeMax = np.sum(info["sizeM"])
    np.set_printoptions(edgeitems = 10)
    captiontxt = f'Network Size: {sizeMax}  K: {info["K"]}  mean_Ext: {info["meanExt"]} \n\
        time: {info["timer"]}   jE: {info["jE"]}   jI: {info["jI"]}' 
    shorttxt   = f'_S{int(np.log10(sizeMax))}'\
                + f'_K{int(np.log10(info["K"]))}_m{str(info["meanExt"])[2:]}_t{str(info["timer"])[:-1]}' # \njE: {jE}   jI: {jI} ' 
    if toDo["doRand"]:
        captiontxt += f",\n stochastic Updates"
        shorttxt += "_rY"
    elif toDo["doDet"]:
        captiontxt += ",\n strictly deterministic Updates"
        shorttxt += "_rNN"
    else:
        captiontxt += ",\n deterministic Updates"
        shorttxt += "_rN"
        
    ### still updating caption and title ###
    if   (toDo["doThresh"] == "constant"): 
        captiontxt += ", Thresholds = constant"
        shorttxt += "_tC"
    elif (toDo["doThresh"] == "gauss"   ):   
        captiontxt += ", Thresholds = gaussian"
        shorttxt += "_tG"
    elif (toDo["doThresh"] == "bound"   ):  
        captiontxt += ", Thresholds = bounded"
        shorttxt += "_tB"

    ### still updating caption and title ###
    figfolder = info['figfolder'] + shorttxt 
    valueFolder = Path(str(info['valueFolder']) + shorttxt)
    if figs:
        plots.figfolder_GLOBAL  = figfolder
        plots.captiontxt_GLOBAL = captiontxt
        plots.titletxt_GLOBAL   = shorttxt
        return [figfolder, shorttxt, captiontxt]
    else:
        return valueFolder

def plot_machine(
        activeOT, fireOT, indiNeuronsDetailed,
        info, drw, toDo
        ):
    threshM, timer, sizeM, recNum = info["threshM"], info["timer"], info["sizeM"], info["recNum"]
    describe(toDo, info,1) 
    ### Analysis ###
    mean_actiOT = analyzeMeanOT(activeOT,sizeM)

    ### Plotting Routine ###
    if drw["pIndiExt"]:
        plots.indiExtended(indiNeuronsDetailed, threshM, recNum )
    if drw["nDistri"]:
        plots.newDistri(activeOT, timer)
    if drw["newMeanOT"]:
        plots.newMeanOT(mean_actiOT)
    if drw["dots2"]:
        plots.dots2(activeOT, timer)
    if drw["nInter_log"]:
        plots.newInterspike(fireOT,timer)
    if drw["nInter"]:
        plots.newInterspike(fireOT,timer,0)
def changeExt_Main():
    info = numParam()
    toDo = doParam()[1]
    info["meanExt"] = 0.04
    jCon = createjCon(info["sizeM"], info["jE"], info["jI"], info["K"])
    external = createExt(info["sizeM"],info["extM"], info["K"], info["meanExt"])     
    thresh = createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
    mE_List= [0.04, 0.1, 0.2, 0.3]
    meanList = []
    for i in range(len(mE_List)):
        external = createExt(info["sizeM"],info["extM"], info["K"],mE_List[i] )     
        activeOT = run_box( jCon, thresh, external,  info, toDo)[1] #1:active, 2:fire
        means= analyzeMeanOT(activeOT,info["sizeM"])
        meanList.append([mE_List[i]])
        meanList[-1] += [np.mean(means[i][10:])for i in range(2)]
    meanList= np.transpose(meanList)
    print(meanList)
    plots.figfolder_GLOBAL = info["figfolder"]
    plots.mean_vs_ext(meanList)

def changeThresh_Main():
    ### Specify Parameters 
    info = numParam()
    (drw, toDo) = doParam()

    ### Create constant inputs to function
    jCon = createjCon(info["sizeM"], info["jE"], info["jI"], info["K"])
    external = createExt(info["sizeM"],info["extM"], info["K"], info["meanExt"])     
    doThresh    = ["constant", "gauss", "bound"]
    for i in range(len(doThresh)):
        toDo["doThresh"] = doThresh[i]
        thresh = createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
        (indiNeuronsDetailed,  
                activeOT, fireOT, nval0
        ) = run_box( jCon, thresh, external, info, toDo,)

        plot_machine(
            activeOT, fireOT, indiNeuronsDetailed,
            info, drw, toDo)

def version_Main():
    ### Specify Parameters 
    info = numParam()
    (drw, toDo) = doParam()

    ### Create constant inputs to function
    jCon = createjCon(info["sizeM"], info["jE"], info["jI"], info["K"])
    external = createExt(info["sizeM"],info["extM"], info["K"], info["meanExt"])     
    thresh = createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
    versions = [[0,0],
                [0,1],
                [1,0]]
    for vers in versions:
        toDo["doRand"],toDo["doDet"] = vers
        (indiNeuronsDetailed, activeOT, fireOT, nval0
        ) =     run_box( jCon, thresh, external, info, toDo,)

        plot_machine(
            activeOT, fireOT, indiNeuronsDetailed,
            info, drw, toDo)



def VanillaMain():
    ### Specify Parameters 
    info = numParam()
    (drw, toDo) = doParam()

    ### Create constant inputs to function
    jCon = createjCon(info["sizeM"], info["jE"], info["jI"], info["K"])
    external = createExt(info["sizeM"],info["extM"], info["K"], info["meanExt"])     
    thresh = createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
    
    #valueFolder = describe(toDo, info,0) 
    (indiNeuronsDetailed,   
            activeOT, fireOT, nval0
    ) = run_box( jCon, thresh, external, info, toDo,)

    plot_machine(
        activeOT, fireOT, indiNeuronsDetailed,
        info, drw, toDo)

###############################################################################
############################# Customize Here ##################################
###############################################################################
def numParam():
    """
    Sets all parameters relevant to the simulation    

    For historic reasons also sets the folder where figures and data are saved
    """
    timestr = time.strftime("%y%m%d_%H%M")
    figfolder = "../figs/testreihe_" + timestr
    valuefoldername = "../ValueVault/testreihe_"
    valueFolder     =  Path(valuefoldername + timestr)
    extM            = np.array([1.,0.8])
    jE              = 2.
    jI              = 1.8
    threshM         = np.array([1., 0.7])
    tau             = 0.9
    meanExt         = 0.1
    meanStartActi   = meanExt
    recNum          = 1
    ### Most changed vars ###
    timer           = 100
    K               = 1000
    size            = 1000
    sizeM           = np.array([size,size])

    info = locals()
    info.pop("valuefoldername")
    info.pop("size")
    return info

def doParam():
    """
    specifies most behaviors of 
    """
    #Bools for if should be peotted or not
    pIndiExt    = 0
    nDistri     = 0
    newMeanOT   = 0
    nInter      = 0
    nInter_log  = 1
    dots2       = 0
    drw = locals()
    
    doThresh    = "constant" #"constant", "gauss", "bound"
    doRand      = 0     #Only one Sequence per Routine
    doDet       = 0

    toDo = {}
    for wrd in ("doThresh", "doRand","doDet"):
        toDo[wrd] = locals()[wrd]

    plots.savefig_GLOBAL    = 1
    plots.showPlots_GLOBAL  = 0
    return drw, toDo

if __name__ == "__main__":
    #changeExt_Main()   
    #changeThresh_Main()
    #version_Main()
    VanillaMain()
    pass
def test():
    class my_pdf(st.rv_continuous):
        def _pdf(self,x, tau):
            return x*math.exp(-x/tau)/(tau**2)  # Normalized over its range, in this case [0,1]

    dist_R = my_pdf(a=0, name='my_pdf')
    x = dist_R(tau = 1)
    print(x)

###############################################################################
################################## To Dos #####################################
###############################################################################
"""
meanExt change what happens, linear read out durch logisitic regression oder pseudo inverse
Trainieren auf entscheidung jetzt, in 1, in 2...
immer 2 Neuronen pro Zeit
input 1 Zeitschritt 1 oder 0 übergeben

"""