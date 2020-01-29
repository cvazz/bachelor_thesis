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
import joblib  # dump,load

### Duh ###
import time
import warnings
### Machine Learning
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import PassiveAggressiveClassifier

### Local Modules ###
import mathsim as msim
import utils
import plots
import old

print_GLOBAL = 1

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
        valueFolder.mkdir(parents=True)

    remember_list = [info, toDo]
    indiNametxt = "indiNeurons"
    infoNametxt = "infoDict"
    fireNametxt = "fireOT"      
    activeNametxt = "activeOT"    

    indiName = utils.makeNewPath(valueFolder, indiNametxt, "npy")
    fireName = utils.makeNewPath(valueFolder, fireNametxt, "npy")
    infoName = utils.makeNewPath(valueFolder, infoNametxt, "pkl")
    activeName = utils.makeNewPath(valueFolder, activeNametxt, "npy")

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

###############################################################################
############################ Creating Functions ###############################
###############################################################################


def createjCon(info):
    """
    Current Connection Matrix Creator (31/10)

    Only working for
    :param     sizeM   : Contains size of exhib and inhib
    :param     jVal    : Contains nonzero Values for Matrix
    :param     K       : Number of connections with inhib/exhib

    :return     jCon    : Connection Matrix
    """
    sizeM, jE, jI, K = info["sizeM"], info["jE"], info["jI"], info["K"]
    j_EE, j_EI = info["j_EE"], info["j_EI"]
    j_IE, j_II = -1*jE, -1*jI
    if print_GLOBAL:
        print("Create jCon")
    timestart = time.time()
    sizeMax     = sizeM[0] + sizeM[1]
    jVal = np.array([[j_EE, j_IE],[j_EI, j_II]])
    jVal = jVal/math.sqrt(K)

    ### Connection Matrix ###

    oddsBeingOne= 2*K/sizeMax
    jCon        = np.random.binomial(1, oddsBeingOne, sizeMax**2)

    jCon        = jCon.astype(float)
    jCon.shape  = (sizeMax, sizeMax)

    ### add weights
    jCon[:sizeM[0],:sizeM[0]] = jCon[:sizeM[0],:sizeM[0]]*jVal[0,0]
    jCon[sizeM[0]:,:sizeM[0]] = jCon[sizeM[0]:,:sizeM[0]]*jVal[1,0]
    jCon[:sizeM[0],sizeM[0]:] = jCon[:sizeM[0],sizeM[0]:]*jVal[0,1]
    jCon[sizeM[0]:,sizeM[0]:] = jCon[sizeM[0]:,sizeM[0]:]*jVal[1,1]

    jCon.setflags(write=False)
    timeend = time.time()
    if print_GLOBAL:
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
        arr = random.sample(arr, len(arr))
        nval += arr
    return np.array(nval)


def createThresh(sizeM, threshM, doThresh):
    if (doThresh == "constant"):
        thresh = createConstantThresh(sizeM, threshM)
    elif (doThresh == "gaussian"):
        thresh = createGaussThresh(sizeM, threshM)
    elif (doThresh == "bounded"):
        thresh = createBoundThresh(sizeM, threshM)
    else:
        raise NameError("Invalid threshold codeword selected")
    thresh.setflags(write=False)
    return thresh


def createConstantThresh(sizeM, threshM):
    """
    Creates Threshold vector with threshold for each Datapoint

    :param      sizeM   : Contains size of exhib and inhib neurons
    :param      threshM : Contains values for threshold
    """
    thresh = []
    for i in range(2):
        thresh.extend([threshM[i] for x in range(sizeM[i])])
    return np.array(thresh)


def createGaussThresh(sizeM, threshM):
    dev = 0.3
    thresh = []
    for i in range(len(sizeM)):
        thresh += [np.random.normal(threshM[i], dev) for x in range(sizeM[i])]
    return np.array(thresh)


def createBoundThresh(sizeM, threshM):
    delta = 0.3
    thresh = []
    for i in range(len(sizeM)):
        thresh += [np.random.uniform(threshM[i]-delta/2,threshM[i]+delta/2)
                   for x in range(sizeM[i])]
    return np.array(thresh)


def createExt(info, meanExt_=""):
    """
    Creates vector of external input for each Datapoint
    (with all exhib and all inhib having the same value)

    :param      sizeM   : Contains size of exhib and inhib neurons
    :param      extM    : Contains factors of external neurons for the inhib
                          values in the system
    :param      K       : Connection Number
    :param      meanExt : Mean activation of external neurons
    """
    sizeM, extM, K = info["sizeM"], info["extM"], info["K"]
    meanExt = info["meanExt"]
    if meanExt_:
        meanExt = meanExt_
    ext = []
    extVal = extM * math.sqrt(K) *meanExt
    for i in range(len(sizeM)):
        ext.extend([extVal[i] for x in range(sizeM[i])])
    external = np.array(ext)
    return external

###############################################################################
############################## Analysis Tools #################################
###############################################################################


def analyzeMeanOT(inputOT, sizeM):
    # Get an array of ints for excitatory and inhibitory,
    # each int represents one activation/firing at this time point
    # int is only one level of precision. It could be modified
    flat = [np.array([x for row in inputOT[:sizeM[0]] for x in row], dtype=int),
            np.array([x for row in inputOT[sizeM[0]:] for x in row], dtype=int)]
    # Count how many individuals have fired at a given time
    uniq = [np.unique(act, return_counts=1) for act in flat]

    # Convert count above into a value between 0 and 1
    normalized = [uniq[i][1]/sizeM[i] for i in range(2)]

    return normalized

###############################################################################
############################## Core Functions #################################
###############################################################################


def timestepMat(iter, nval, jCon, thresh, external,
                recordPrecisely=0, combMinSize=[0], combMaxSize=[0]):
    """
    Calculator for whether one neuron changes value

    Sums all the input with corresponding weights.
    Afterwords adds external input and subtracts threshold.
    Result is plugged in Heaviside function

    :param      iter    : iterator, determines which neuron is to be changed
    :param      nval    : current values of all neurons, is CHANGED to reflect 
                          new value within function
    :param      jCon    : Connection Matrix
    :param      thresh  : Stores Thresholds
    :param      external: Input from external Neurons

    :return             : for troubleshooting returns value 
                          before Heaviside function
    """
    sum = jCon[iter].dot(nval)
    decide = sum + external[iter] - thresh[iter]
    nval[iter] = int(decide > 0)
    if recordPrecisely:
        inputs = []
        for i in range(len(combMinSize)):
            inputs.append(jCon[iter,combMinSize[i]:combMaxSize[i]].
                          dot(nval[combMinSize[i]:combMaxSize[i]]))
        return [inputs[0]+external[iter], (decide + thresh[iter]),
                inputs[1], inputs[0], external[iter]]
    return decide


# def update_org_OLD(nval, jCon, thresh, external,
#                    indiNeuronsDetailed, randomProcess, info):
#     """
#     Selects the sequence of updates and records results on the fly

#     Randomly chooses between excitatory or inhibitory sequence with relative likelihood tau
#     to choose inhibitory (ie 1 meaning equally likely).
#     Each round a new permutation of range is drawn
#     Currently only supports recording individual excitatory neurons for indiNeuronsDetailed


#     :param      maxTime : Controls runtime
#     :param      sizeM   : Contains information over the network size
#     :param      tau     : How often inhibitory neurons fire compared to excitatory
#     :param      nval    : current values of all neurons, is CHANGED to reflect new value within function
#     :param      jCon    : Connection Matrix 
#     :param      thresh  : Stores Thresholds 
#     :param      external: Input from external Neurons 
#     :param      indiNeuronsDetailed: 
#     :param      recNum  : How many neurons are recorded 

#     :return     nvalOvertime
#     """



#     (sizeM, maxTime, tau, recNum) = info["sizeM"],info["timer"], info["tau"], info['recNum']
#     #1
#     sizeMax = sum(sizeM)    
#     likelihood_of_choosing_excite =  tau / (1+tau)

#     ### New record containers ###
#     activeOT = [[] for _ in range(sizeMax)]
#     fireOT   = [[] for _ in range(sizeMax)]

#     comb_Big_Time   = [0, 0]        # is added up maxTime times before hard stop
#     comb_Small_Time = [0, 0]        # is added up N times before being resetted 
#     combMinSize     = np.array([0, sizeM[0]])
#     combMaxSize     = combMinSize + sizeM
#     combRange       = [np.arange(combMinSize[i],combMaxSize[i]) for i in range(2)]
#     combSequence    = []

#     for inhibite in range(2):
#         if randomProcess: 
#             combSequence.append(np.random.randint(
#                     combMinSize[inhibite],combMaxSize[inhibite], sizeM[inhibite]))
#         else:
#             combSequence.append( np.random.permutation(combRange[inhibite]))
#     #2

#     while comb_Big_Time[0] < maxTime:
#         # inhibite = 0 with likelihood of choosing excite
#         inhibite = int(np.random.uniform(0,1)>likelihood_of_choosing_excite)
#         # chooses the next neuron to be iterated through
#         iterator = combSequence[inhibite][comb_Small_Time[inhibite]]
#         # records the first "recNum" values
#         recordPrecisely = iterator <recNum
#         # checks whether the neuron was just active
#         justActive = nval[iterator] 

#         result = timestepMat(iterator, nval, jCon,
#                 thresh, external,  recordPrecisely,
#                 combMinSize, combMaxSize)
#         ### if result is of type list it needs to be recorded ...
#         if isinstance(result, list):
#             indiNeuronsDetailed[iterator].append(result)
#             # ... and converted back to a float value
#             result = result[1] - thresh[iterator]
#         ### Record Activation
#         if result >= 0:
#             temp = comb_Big_Time[0]+comb_Small_Time[0]/sizeM[0]
#             activeOT[iterator].append(temp)
#             if not justActive:
#                 fireOT[iterator].append(temp)

#         comb_Small_Time[inhibite] +=1
#         ### End of comb_Small_Time Sequence
#         if comb_Small_Time[inhibite] >= sizeM[inhibite]:
#             comb_Big_Time[inhibite] +=1
#             comb_Small_Time[inhibite] = 0
#             #3
#             if randomProcess: 
#                 combSequence[inhibite]  = np.random.randint(
#                     combMinSize[inhibite],combMaxSize[inhibite], sizeM[inhibite])
#             else:
#                 combSequence[inhibite] = np.random.permutation(combRange[inhibite])
#             if comb_Big_Time[0] % 10 == 0 and not inhibite:
#                 print(f"{(comb_Big_Time[0]/maxTime):.0%}", end=", ", flush=True)
#     print("")
    
#     return  activeOT, fireOT, #4


# def transform2binary(xOT,timer):
#     """Transforms information of spike times into binary record of spikes

#     Top-level is time or in other words a list of spikes at time i. The index of a one
#     marks the location of the spike
#     :param xOT: [description]
#     :type xOT: [type]
#     :param timer: [description]
#     :type timer: [type]
#     :return: [description]
#     :rtype: [type]
#     """
#     intOT = [np.trunc(row) for row in xOT]
#     output = []
#     for row in intOT:
#         output.append([])
#         for i in range(timer):
#             output[-1].append(int(i in row))
#     return np.transpose(output)


def update_org( nval, jCon, thresh, external,
                indiNeuronsDetailed, toDo, info):
    """
    Selects the sequence of updates and records results on the fly

    Randomly chooses between excitatory or inhibitory sequence with relative likelihood tau 
    to choose inhibitory (ie 1 meaning equally likely).
    Each round a new permutation of range is drawn


    :param      nval    : current values of all neurons, is CHANGED to reflect
                          new value within function 
    :param      jCon    : Connection Matrix 
    :param      thresh  : Stores Thresholds 
    :param      external: Input from external Neurons 
    :param      indiNeuronsDetailed: 
    :param      toDo    : Contains all specifications about how the program 
                          should run defined in numParam()
    :param      info    : Contains all numeric parameters defined in numParam()

    :return     nvalOvertime
    """
    (sizeM, maxTime, tau, recNum) = info["sizeM"],info["timer"], info["tau"], info['recNum']
    switch, randomProcess, meanExt_M = toDo["switch"], toDo["doRand"], info["meanExt_M"] 
    sizeMax = sum(sizeM)    
    likelihood_of_choosing_excite =  tau / (1+tau)

    labels = list(np.random.binomial(1,.5,maxTime))
    if switch:
        external = createExt(info,meanExt_M[labels[0]])

    ### Information Recorders ###
    activeOT = [[] for _ in range(sizeMax)]
    fireOT   = [[] for _ in range(sizeMax)]

    ### Defining Loop Parameters 
    # First position is excitatory, second is inhibitory
    comb_Big_Time   = [0, 0]        # is added up maxTime times before hard stop
    comb_Small_Time = [0, 0]        # is added up N times before being resetted 
    combMinSize     = np.array([0, sizeM[0]])
    combMaxSize     = combMinSize + sizeM
    combRange       = [np.arange(combMinSize[i],combMaxSize[i]) for i in range(2)]
    combSequence    = []
    ### Choose Update Sequence 
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

        ### Calculate next Step ###
        result = timestepMat(iterator, nval, jCon,
                thresh, external,  recordPrecisely,
                combMinSize, combMaxSize)

        # if result is of type list it needs to be recorded ...
        if isinstance(result, list):
            indiNeuronsDetailed[iterator].append(result)
        # ... and converted back to a float value
            result = result[1] - thresh[iterator]

        ### Record Activation ###
        if result >= 0:
            # Calculate The time it was recorded
            temp = comb_Big_Time[0]+comb_Small_Time[0]/sizeM[0]
            activeOT[iterator].append(temp)
            if not justActive:
                fireOT[iterator].append(temp)

        comb_Small_Time[inhibite] +=1

        ### End of comb_Small_Time Sequence
        if comb_Small_Time[inhibite] >= sizeM[inhibite]:
            # Reset Inner Loop
            comb_Small_Time[inhibite] = 0
            # Tick Up Outer Loop
            comb_Big_Time[inhibite] +=1

            ### Checks whether to update external input function ###
            # 1. Did label change compared to last run, 2. is Switch Active,
            # 3. Is this an excitatory process? 4. Is this the last run already?
            if (switch and inhibite == 0 and comb_Big_Time[0] < maxTime and
                not (labels[comb_Big_Time[0]] == labels[comb_Big_Time[0]-1])):  
                # Update External Input
                external = createExt(info, meanExt_M[labels[comb_Big_Time[0]]])

            ### Update the sequence of Updates ###
            if randomProcess: 
                combSequence[inhibite]  = np.random.randint(
                    combMinSize[inhibite],combMaxSize[inhibite], sizeM[inhibite])
            else:
                combSequence[inhibite] = np.random.permutation(combRange[inhibite])

            ### Print ### 
            if comb_Big_Time[0] % 10 == 0 and not inhibite and print_GLOBAL:
                print(f"{(comb_Big_Time[0]/maxTime):.0%}", end=", ", flush=True)
                # if GUI:

    print("")
    return  activeOT, fireOT, labels

###############################################################################
########################### Setup Functions ###################################
###############################################################################


# def abstand(x,y,max_):
#     xx = []
#     yy = []
#     max_delay = max_ - 1
#     for i in range(max_delay):
#         xx.append(x[i:-max_delay+i])
#         yy.append(y[:-max_delay])
#     xx.append(x[max_delay:])
#     yy.append(y[:-max_delay])
#     return xx, yy


# def comp(label, estimate):
#     accu = np.zeros((4),  np.int64)

#     t = np.zeros((2,2))
#     for i in range(len(label)):
#         pos = label[i] + estimate[i]*2
#         accu[0] += label[i] and estimate[i]
#         accu[1] += (not label[i]) and estimate[i]
#         accu[2] += label[i] and not estimate[i]
#         accu[3] += not label[i] and not estimate[i]
#         t[label[i]][estimate[i]] += 1
#     total = np.sum(accu)
#     accu_str = ( "true positive", "false_alarm", "miss \t", "true negative",)
#     if print_GLOBAL:
#         for i in range(len(accu)):
#             print(accu_str[i] + ":\t" + str(accu[i]/total))
#     return accu


# # def run_wonder( jCon, thresh, external, info, toDo):
# #     """
# #     executes the differen sequences
# #     """ 
# #     print("run")
# #     timestart = time.time()
# #     valueFolder = describe( toDo, info, 0 )

# #     ### The function ###
# #     nval = createNval(info["sizeM"], info["meanStartActi"])  
# #     indiNeuronsDetailed = [[] for i in range(info['recNum'])] 
# #     activeOT, fireOT, labels = update_org( 
# #         nval, jCon, thresh, external, indiNeuronsDetailed, toDo, info)

# #     ### time check ###
# #     timeend = time.time()
# #     if print_GLOBAL:
# #         print("runtime of routine")
# #         utils.timeOut(timeend - timestart)

# #     saveResults(valueFolder, indiNeuronsDetailed, 
# #                 activeOT, fireOT, info, toDo)
# #     # (indiNeuronsDetailed, activeOT, fireOT, nval_OT, info, toDo
# #     # )= recoverResults(valueFolder)

# #     return (indiNeuronsDetailed,   
# #             activeOT, fireOT, labels)


# def run_box( jCon, thresh, external, info, toDo ):
#     """
#     executes the differen sequences
#     """ 
#     # info['timestr']
#     if print_GLOBAL:
#         print("run")
#     timestart = time.time()
#     valueFolder = describe( toDo, info, 0 )

#     ### The function ###
#     nval = createNval(info["sizeM"], info["meanStartActi"])  
#     indiNeuronsDetailed = [[] for i in range(info['recNum'])] 
#     activeOT, fireOT, labels = update_org( nval, jCon, thresh, external,
#                                 indiNeuronsDetailed, toDo, info)

#     ### time check ###
#     timeend = time.time()
#     if print_GLOBAL:
#         print("runtime of routine")
#         utils.timeOut(timeend - timestart)
#     if not toDo["switch"]:
#         saveResults(valueFolder, indiNeuronsDetailed, 
#                     activeOT, fireOT, info, toDo)
#     # (indiNeuronsDetailed, activeOT, fireOT, nval_OT, info, toDo
#     # )= recoverResults(valueFolder)

#     return (indiNeuronsDetailed,   
#             activeOT, fireOT, labels)

# def describe( toDo, info, figs ):

#     sizeMax = np.sum(info["sizeM"])
#     np.set_printoptions(edgeitems = 10)
#     captiontxt = f'Network Size: {sizeMax}  K: {info["K"]}  mean_Ext: {info["meanExt"]} \n\
#         time: {info["timer"]}   jE: {info["jE"]}   jI: {info["jI"]}\n j_EE: {np.round(info["j_EE"],3)}, ext_E: {np.round(info["extM"][0])}' 
#     shorttxt   = f'j_EE_{str(info["j_EE"])[:3]}_ext_E_{str(info["extM"][0])[:3]}' 

#         # f'_S{int(np.log10(sizeMax))}'\
#                 # + f'_K{int(np.log10(info["K"]))}_m{str(info["meanExt"])[2:]}_t{str(info["timer"])[:-1]}' # \njE: {jE}   jI: {jI} ' 

#     if toDo["doRand"]:
#         captiontxt += f",\n stochastic Updates"
#         shorttxt += "_rY"
#     else:
#         captiontxt += ",\n deterministic Updates"
#         shorttxt += "_rN"

#     if   (toDo["doThresh"] == "constant"): 
#         captiontxt += ", Thresholds = constant"
#         shorttxt += "_tC"
#     elif (toDo["doThresh"] == "gauss"   ):   
#         captiontxt += ", Thresholds = gaussian"
#         shorttxt += "_tG"
#     elif (toDo["doThresh"] == "bound"   ):  
#         captiontxt += ", Thresholds = bounded"
#         shorttxt += "_tB"

#     ### still updating caption and title ###
#     figfolder = info['figfolder'] + shorttxt 
#     valueFolder = Path(str(info['valueFolder']) + shorttxt)
#     if figs:
#         plots.figfolder_GLOBAL  = figfolder
#         plots.captiontxt_GLOBAL = captiontxt
#         plots.titletxt_GLOBAL   = shorttxt
#         return [figfolder, shorttxt, captiontxt]
#     else:
#         return valueFolder

# def plot_machine(
#         activeOT, fireOT, indiNeuronsDetailed,
#         info, drw, toDo
#         ):
#     threshM, timer, sizeM, recNum = info["threshM"], info["timer"], info["sizeM"], info["recNum"]
#     describe(toDo, info,1) 
#     ### Analysis ###
#     mean_actiOT = analyzeMeanOT(activeOT,sizeM)

#     ### Plotting Routine ###
#     if drw["pIndiExt"]:
#         plots.indiExtended(indiNeuronsDetailed, threshM, recNum )
#     if drw["nDistri"]:
#         plots.newDistri(activeOT, timer)
#     if drw["newMeanOT"]:
#         plots.newMeanOT(mean_actiOT)
#     if drw["dots2"]:
#         plots.dots2(activeOT, timer)
#     if drw["nInter_log"]:
#         plots.newInterspike(fireOT,timer)
#     if drw["nInter"]:
#         plots.newInterspike(fireOT,timer,0)

# def changeExt_Main():
#     info = numParam()
#     toDo = doParam()[1]
#     info["meanExt"] = 0.04
#     jCon = createjCon(info["sizeM"], info["jE"], info["jI"], info["K"])
#     external = createExt(info["sizeM"],info["extM"], info["K"], info["meanExt"])     
#     thresh = createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
#     mE_List= [0.04, 0.1, 0.2, 0.3]
#     meanList = []
#     for i in range(len(mE_List)):
#         external = createExt(info["sizeM"],info["extM"], info["K"],mE_List[i] )     
#         activeOT = run_box( jCon, thresh, external,  info, toDo)[1] #1:active, 2:fire
#         means= analyzeMeanOT(activeOT,info["sizeM"])
#         meanList.append([mE_List[i]])
#         meanList[-1] += [np.mean(means[i][10:])for i in range(2)]
#     meanList= np.transpose(meanList)
#     print(meanList)
#     plots.figfolder_GLOBAL = info["figfolder"]
#     plots.mean_vs_ext(meanList)

# def changeThresh_Main():
#     ### Specify Parameters 
#     info = numParam()
#     (drw, toDo) = doParam()

#     ### Create constant inputs to function
#     jCon = createjCon(info["sizeM"], info["jE"], info["jI"], info["K"])
#     external = createExt(info["sizeM"],info["extM"], info["K"], info["meanExt"])     
#     doThresh    = ["constant", "gauss", "bound"]
#     for i in range(len(doThresh)):
#         toDo["doThresh"] = doThresh[i]
#         thresh = createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
#         (indiNeuronsDetailed,  
#                 activeOT, fireOT, _ #label
#         ) = run_box( jCon, thresh, external, info, toDo,)

#         plot_machine(
#             activeOT, fireOT, indiNeuronsDetailed,
#             info, drw, toDo)
# def split_train_test(x,y,split_point):
#     """ Splits two arrays of size (N x rows x beliebig) in roughly half
    
#     :param x: [description]
#     :type x: [type]
#     :param y: [description]
#     :type y: [type]
#     :param split_point: [description]
#     :type split_point: [type]
#     :return: [description]
#     :rtype: [type]
#     """
#     split = int(len(x[0])* split_point)
#     x_a, y_a, x_b, y_b = [],[],[],[]
#     for x_spalte,y_spalte in zip(x,y):
#         x_a.append(x_spalte[:split])
#         x_b.append(x_spalte[split:])
#         y_a.append(y_spalte[:split])
#         y_b.append(y_spalte[split:])
        
#     return x_a, y_a, x_b, y_b

# def MachineMain(jEE = 1,extE = 1):
#     ### Specify Parameters 
#     info = numParam()
#     drw,toDo = doParam()
#     train = 1
#     test  = 1
#     dist = 3
#     toDo["switch"] = 1
#     info["j_EE"] = jEE
#     info["extM"][0] = extE
#     ### Create constant inputs to function
#     jCon = createjCon(info)
#     external = createExt(info["sizeM"],info["extM"], info["K"], info["meanExt"])     
#     thresh = createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
    
#     modelfolder =  utils.checkFolder("../ML_Model")
#     general_name = "logreg_"
#     modelname = modelfolder + general_name + info["timestr"] 
#     # modelname = modelfolder + general_name + "200105_1829"
#     (indiNeuronsDetailed,   
#             activeOT_train, fireOT, labels_train
#     ) = run_box( jCon, thresh, external, info, toDo,)

#     ### Learning Part
#     input_train= transform2binary(activeOT_train, info["timer"])
#     model_A = [LogisticRegression(solver = "lbfgs", max_iter = 160) for _ in range(dist)]
#     x_A, y_A = abstand(input_train,labels_train,dist)
#     xtrain_A, ytrain_A,xtest_A, ytest_A = split_train_test(x_A, y_A,0.5)

#     ### Test ###
#     for i in range(dist):
#         model_A[i].fit(xtrain_A[i], ytrain_A[i])
#     estimate_A = [model_A[i].predict(xtest_A[i]) for i in range(dist)]
#     readout_ = [comp(ytest_A[i],estimate_A[i]) for i in range(dist)]

#     ### Convert Readout to single precision value ###
#     correctness = [(delay[0]+delay[3])/sum(delay) for delay in readout_]
#     # print("correctness") 
#     # print(correctness) 
#     # plot_machine(
#     #     activeOT_test, fireOT, indiNeuronsDetailed,
#     #     info, drw, toDo)
#     useless(indiNeuronsDetailed,fireOT)
#     return correctness

# def test_the_machine():
     
#     plots.print_GLOBAL = 0
#     global print_GLOBAL 
#     print_GLOBAL = 0
    
#     timestart = time.time()
#     readout_OT = []
#     record_warnings = []
#     with warnings.catch_warnings(record=True) as warn_me:
#         last_warn = 0
#         for _ in range(5):
#             readout = []
#             # for extE in np.linspace(0.7,1.,2):
#                 # for jEE in np.linspace(1., 1.5, 2):
#             for extE in np.linspace(0.6,1.2,8):
#                 for jEE in np.linspace(.8, 1.7, 8):
#                     print(f"jEE: {np.round(jEE,2)}, extE: {np.round(extE,2)}")
#                     try:

#                         readout.append([jEE, extE, *MachineMain(jEE,extE)])
#                         if warn_me and warn_me != last_warn:
#                             print("this is the warning at: ",end="")
#                             print(extE, end=", ")
#                             print(jEE)
#                             record_warnings.append(warn_me[-1])
#                             print(warn_me[-1].category)
#                             print(warn_me[-1].message)

#                     except:
#                         print(readout)
#                         # np.save("test_the_machine2",readout) 
#                         timeend = time.time()
#                         utils.timeOut(timeend - timestart)
#                         raise
#         readout_OT.append(readout) 
#     print(record_warnings)
#     np.save("test_the_machineOT",readout_OT) 
#     timeend = time.time()
#     utils.timeOut(timeend - timestart)

# def VanillaMain():
#     ### Specify Parameters 
#     info = numParam()
#     (drw, toDo) = doParam()
#     toDo["switch"] = 0
#     ### Create constant inputs to function
#     jCon = createjCon(info["sizeM"], info["jE"], info["jI"], info["K"])
#     external = createExt(info["sizeM"],info["extM"], info["K"], info["meanExt"])     
#     thresh = createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
    
#     #valueFolder = describe(toDo, info,0) 
#     (indiNeuronsDetailed,   
#             activeOT, fireOT, label
#     ) = run_box( jCon, thresh, external, info, toDo,)

#     plot_machine(
#         activeOT, fireOT, indiNeuronsDetailed,
#         info, drw, toDo)
#     print(np.shape(indiNeuronsDetailed))
#     ext_contrib = np.sum(indiNeuronsDetailed[0][:][4])
#     int_contrib = np.sum(indiNeuronsDetailed[0][:][3])
#     print(int_contrib/ext_contrib)
#     useless(label)

# ###############################################################################
# ############################# Customize Here ##################################
# ###############################################################################
# def setupFolder():
#     timestr = time.strftime("%y%m%d_%H%M")
#     figfolder = "../figs/testreihe_" + timestr
#     valuefoldername = "../ValueVault/testreihe_"
#     valueFolder     =  Path(valuefoldername + timestr)
#     return timestr, figfolder, valueFolder

# def numParam():
#     """
#     Sets all parameters relevant to the simulation    

#     For historic reasons also sets the folder where figures and data are saved
#     """

#     timestr, figfolder, valueFolder = setupFolder()
#     j_EE            = 1
#     j_EI            = 1
#     extM            = np.array([1,0.8])
#     jE              = 2.
#     jI              = 1.8
#     threshM         = np.array([1., 0.7])
#     tau             = 0.9
#     meanExt         = 0.1
#     meanStartActi   = meanExt
#     recNum          = 1
#     ### Most changed vars ###
#     timer           = 220
#     K               = 1000
#     size            = 4000
#     sizeM           = np.array([size,size])

#     info = locals()
#     info["meanExt_M"] = [.1, .3]
#     # info["GUI"] = 0
#     info.pop("size")
#     return info


# def doParam():
#     """
#     specifies most behaviors of 
#     """
#     #Bools for if should be peotted or not
#     pIndiExt    = 1
#     nDistri     = 1
#     newMeanOT   = 1
#     nInter      = 0
#     nInter_log  = 0
#     dots2       = 1
#     drw = locals()
    
#     doThresh    = "constant" #"constant", "gaussian", "bounded"
#     switch      = 0     #change external input?
#     doRand      = 0     #Only one Sequence per Routine

#     toDo = {}
#     for wrd in ("doThresh", "doRand","switch"):
#         toDo[wrd] = locals()[wrd]

#     plots.savefig_GLOBAL    = 1
#     plots.showPlots_GLOBAL  = 0
#     return drw, toDo

# def useless(*args):
#     pass
    
# def check_plt():
#     plt.figure()
# if __name__ == "__main__":
#     check_plt()
#     # changeExt_Main()   
#     # changeThresh_Main()
#     # VanillaMain()
#     # test_the_machine()
#     MachineMain()
#     # testXX()
#     pass

# ###############################################################################
# ################################## To Dos #####################################
# ###############################################################################
# """
# meanExt change what happens, linear read out durch logisitic regression oder pseudo inverse
# Trainieren auf entscheidung jetzt, in 1, in 2...
# immer 2 Neuronen pro Zeit
# input 1 Zeitschritt 1 oder 0 übergeben
# Plots: IndiNeuronPlot: untere Linie Peaks nicth korrekt
# Plots: Distri Fit

# Statistik 5 mal wiederholen
# Sanity Check
# Prozess optimieren
# mehrere Epochs
# Input Verhältnis Excitatory zu Extern
# """