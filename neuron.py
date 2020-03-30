###############################################################################
############################## Imported Modules ###############################
###############################################################################

### Numbers ###
import numpy as np
import math
import random
# import scipy.stats as st

# import matplotlib.pyplot as plt
### File Interaction and Manipulation ###
# from pathlib import Path
import pickle
from types import SimpleNamespace
# import joblib  # dump,load

### Duh ###
import time
# import warnings
### Machine Learning
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import PassiveAggressiveClassifier

### Local Modules ###
# import mathsim as msim
import utils
# import plots
# import old

print_GLOBAL = 1

###############################################################################
############################## Utility Functions ##############################
###############################################################################

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
    j_IE, j_II = -1 * jE, -1 * jI
    if print_GLOBAL:
        print("Create jCon")
    timestart = time.time()
    sizeMax     = sizeM[0] + sizeM[1]
    jVal = np.array([[j_EE, j_IE], [j_EI, j_II]])
    jVal = jVal / math.sqrt(K)

    ### Connection Matrix ###

    oddsBeingOne = 2 * K / sizeMax
    jCon        = np.random.binomial(1, oddsBeingOne, sizeMax**2)

    jCon        = jCon.astype(float)
    jCon.shape  = (sizeMax, sizeMax)

    ### add weights
    jCon[:sizeM[0], :sizeM[0]] = jCon[:sizeM[0], :sizeM[0]] * jVal[0, 0]
    jCon[sizeM[0]:, :sizeM[0]] = jCon[sizeM[0]:, :sizeM[0]] * jVal[1, 0]
    jCon[:sizeM[0], sizeM[0]:] = jCon[:sizeM[0], sizeM[0]:] * jVal[0, 1]
    jCon[sizeM[0]:, sizeM[0]:] = jCon[sizeM[0]:, sizeM[0]:] * jVal[1, 1]

    jCon.setflags(write=False)
    timeend = time.time()
    if print_GLOBAL:
        utils.timeOut(timeend - timestart)
    return jCon


def createNval(sizeM, activeAtStart):
    """
    Initializes neuron values "nval" with starting values

    :param      sizeM   : Contains size of exhib and inhib neurons
    :param      extM    : Contains factors of external neurons for the inhib
                         values in the system
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
        thresh += [np.random.uniform(threshM[i] - delta / 2,
                                     threshM[i] + delta / 2)
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
    extVal = extM * math.sqrt(K) * meanExt
    for i in range(len(sizeM)):
        ext.extend([extVal[i] for x in range(sizeM[i])])
    external = np.array(ext)
    return external


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
            inputs.append(jCon[iter, combMinSize[i]:combMaxSize[i]].
                          dot(nval[combMinSize[i]:combMaxSize[i]]))
        return [inputs[0] + external[iter], (decide + thresh[iter]),
                inputs[1], inputs[0], external[iter], thresh[iter]]
    return decide


# def update_org( nval, jCon, thresh, external,
#                 indiNeuronsDetailed, toDo, info):
#     """
#     Selects the sequence of updates and records results on the fly

#     Randomly chooses between excitatory or inhibitory sequence with relative 
# likelihood tau 
#     to choose inhibitory (ie 1 meaning equally likely).
#     Each round a new permutation of range is drawn


#     :param      nval    : current values of all neurons, is CHANGED to reflect
#                           new value within function 
#     :param      jCon    : Connection Matrix 
#     :param      thresh  : Stores Thresholds 
#     :param      external: Input from external Neurons 
#     :param      indiNeuronsDetailed: 
#     :param      toDo    : Contains all specifications about how the program 
#                           should run defined in numParam()
#     :param      info    : Contains all numeric parameters defined in numParam()

#     :return     nvalOvertime
#     """
#     (sizeM, maxTime, tau, recNum) = info["sizeM"],info["timer"], info["tau"], info['recNum']
#     switch, randomProcess, meanExt_M = toDo["switch"], toDo["doRand"], info["meanExt_M"] 
#     sizeMax = sum(sizeM)    
#     likelihood_of_choosing_excite =  tau / (1+tau)



#     labels = list(np.random.binomial(1,.5,maxTime))
#     if switch:
#         external = createExt(info,meanExt_M[labels[0]])

#     ### Information Recorders ###
#     activeOT = [[] for _ in range(sizeMax)]
#     fireOT   = [[] for _ in range(sizeMax)]

#     ### Defining Loop Parameters 
#     # First position is excitatory, second is inhibitory
#     comb_Big_Time   = [0, 0]        # is added up maxTime times before hard stop
#     comb_Small_Time = [0, 0]        # is added up N times before being resetted 
#     combMinSize     = np.array([0, sizeM[0]])
#     combMaxSize     = combMinSize + sizeM
#     combRange       = [np.arange(combMinSize[i],combMaxSize[i]) for i in range(2)]
#     combSequence    = []
#     ### Choose Update Sequence 
#     for inhibite in range(2):
#         if randomProcess: 
#             combSequence.append(np.random.randint(
#                     combMinSize[inhibite],combMaxSize[inhibite], sizeM[inhibite]))
#         else:
#             combSequence.append( np.random.permutation(combRange[inhibite]))

#     while comb_Big_Time[0] < maxTime:
#         # inhibite = 0 with likelihood of choosing excite
#         inhibite = int(np.random.uniform(0,1)>likelihood_of_choosing_excite)
#         # chooses the next neuron to be iterated through
#         iterator = combSequence[inhibite][comb_Small_Time[inhibite]]
#         # records the first "recNum" values
#         recordPrecisely = iterator <recNum
#         # checks whether the neuron was just active
#         justActive = nval[iterator] 

#         ### Calculate next Step ###
#         result = timestepMat(iterator, nval, jCon,
#                 thresh, external,  recordPrecisely,
#                 combMinSize, combMaxSize)

#         # if result is of type list it needs to be recorded ...
#         if isinstance(result, list):
#             indiNeuronsDetailed[iterator].append(result)
#         # ... and converted back to a float value
#             result = result[1] - thresh[iterator]

#         ### Record Activation ###
#         if result >= 0:
#             # Calculate The time it was recorded
#             temp = comb_Big_Time[0]+comb_Small_Time[0]/sizeM[0]
#             activeOT[iterator].append(temp)
#             if not justActive:
#                 fireOT[iterator].append(temp)

#         comb_Small_Time[inhibite] +=1

#         ### End of comb_Small_Time Sequence
#         if comb_Small_Time[inhibite] >= sizeM[inhibite]:
#             # Reset Inner Loop
#             comb_Small_Time[inhibite] = 0
#             # Tick Up Outer Loop
#             comb_Big_Time[inhibite] +=1

#             ### Checks whether to update external input function ###
#             # 1. Did label change compared to last run, 2. is Switch Active,
#             # 3. Is this an excitatory process? 4. Is this the last run already?
#             if (switch and inhibite == 0 and comb_Big_Time[0] < maxTime and
#                 not (labels[comb_Big_Time[0]] == labels[comb_Big_Time[0]-1])):  
#                 # Update External Input
#                 external = createExt(info, meanExt_M[labels[comb_Big_Time[0]]])

#             ### Update the sequence of Updates ###
#             if randomProcess: 
#                 combSequence[inhibite]  = np.random.randint(
#                     combMinSize[inhibite],combMaxSize[inhibite], sizeM[inhibite])
#             else:
#                 combSequence[inhibite] = np.random.permutation(combRange[inhibite])

#             ### Print ### 
#             if comb_Big_Time[0] % 10 == 0 and not inhibite and print_GLOBAL:
#                 print(f"{(comb_Big_Time[0]/maxTime):.0%}", end=", ", flush=True)
#                 # if GUI:
#     print("")

#     return  activeOT, fireOT, labels

def select_update_mechanism(toDo):
    update_mode = toDo["update_mode"]
    if update_mode == "count_up":
        mono_increase = 1
        doStatic = 1
        doRand = 0
    elif update_mode == "static":
        mono_increase = 0
        doStatic = 1
        doRand = 0
    elif update_mode == "permute":
        mono_increase = 0
        doStatic = 0
        doRand = 0
    elif update_mode == "rand":
        mono_increase = 0
        doStatic = 0
        doRand = 1
    return mono_increase, doStatic, doRand    

def run_update(jCon, thresh, external, info, toDo):
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
    mono_increase, doStatic, randomProcess = select_update_mechanism(toDo)
    sizeMax = sum(sizeM)    
    likelihood_of_choosing_excite =  tau / (1+tau)


    nval = createNval(info["sizeM"], info["meanStartActi"]) 
    if "flip" in info:
        flip = info["flip"]
        positions = random.sample(list(np.arange(len(nval))), flip)
        for i in positions:
            nval[i] = abs(nval[i] -1)
    indiNeuronsDetailed = [[] for i in range(2*info['recNum'])]
    # indiNeuronsDetailed = [[] for i in range(info['recNum'])]

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
    print_steps = .1
    print_count = print_steps * maxTime
    ### Choose Update Sequence 
    if mono_increase:
        combSequence = combRange
    else:
        for inhibite in range(2):
            if randomProcess: 
                combSequence.append(np.random.randint(
                    combMinSize[inhibite], combMaxSize[inhibite], sizeM[inhibite]
                ))
            else:
                combSequence.append(np.random.permutation(combRange[inhibite]))
 
    while comb_Big_Time[0] < maxTime:
        # inhibite = 0 with likelihood of choosing excite
        inhibite = int(np.random.uniform(0,1)>likelihood_of_choosing_excite)
        # chooses the next neuron to be iterated through
        iterator = combSequence[inhibite][comb_Small_Time[inhibite]]
        # records the first "recNum" values
        # Original Version, records only excite: 
        recordPrecisely = comb_Small_Time[inhibite] <recNum
        # recordPrecisely = iterator <recNum
        # checks whether the neuron was just active
        justActive = nval[iterator] 

        ### Calculate next Step ###
        result = timestepMat(iterator, nval, jCon,
                thresh, external,  recordPrecisely,
                combMinSize, combMaxSize)

        # if result is of type list it needs to be recorded ...
        if isinstance(result, list) :
            # if iterator > recNum: 
            it = (comb_Small_Time[inhibite] + recNum
                  if inhibite else comb_Small_Time[inhibite])
            indiNeuronsDetailed[it].append(result)
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
                if not doStatic:
                    combSequence[inhibite] = np.random.permutation(combRange[inhibite])

            ### Print ### 
            # if comb_Big_Time[0] % 10 == 0 and not inhibite and print_GLOBAL:
            #     print(f"{(comb_Big_Time[0]/maxTime):.0%}", end=", ", flush=True)
            #     # if GUI:
            if comb_Big_Time[0] >= print_count and not inhibite and print_GLOBAL:
                while comb_Big_Time[0] >= print_count:
                    print(int((print_count/maxTime)*100),end='%, ', flush=True)
                    print_count+=print_steps*maxTime
                # print(f"{(comb_Big_Time[0]/maxTime):.0%}", end=", ", flush=True)

    print("")
    return  indiNeuronsDetailed, activeOT, fireOT, labels

def setup(jCon, thresh, external, info, toDo):
    (sizeM, maxTime, tau, recNum) = info["sizeM"],info["timer"], info["tau"], info['recNum']
    switch, randomProcess, meanExt_M = toDo["switch"], toDo["doRand"], info["meanExt_M"] 
    sizeMax = sum(sizeM)    
    likelihood_of_choosing_excite =  tau / (1+tau)

    
    nval = createNval(info["sizeM"], info["meanStartActi"]) 
    indiNeuronsDetailed = [[] for i in range(2*info['recNum'])]
    # indiNeuronsDetailed = [[] for i in range(info['recNum'])]

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
    
    return SimpleNamespace(**locals())

def inside_loop(hn):
    same = hn.comb_Big_Time[0]
    while same == hn.comb_Big_Time[0]:
        # inhibite = 0 with likelihood of choosing excite
        inhibite = int(np.random.uniform(0,1)>hn.likelihood_of_choosing_excite)
        # chooses the next neuron to be iterated through
        iterator = hn.combSequence[inhibite][hn.comb_Small_Time[inhibite]]
        # records the first "recNum" values
        # Original Version, records only excite: 
        recordPrecisely = hn.comb_Small_Time[inhibite] <hn.recNum
        # recordPrecisely = iterator <recNum
        # checks whether the neuron was just active
        justActive = hn.nval[iterator] 

        ### Calculate next Step ###
        result = timestepMat(iterator, hn.nval, hn.jCon,
                hn.thresh, hn.external,  recordPrecisely,
                hn.combMinSize, hn.combMaxSize)

        # if result is of type list it needs to be recorded ...
        if isinstance(result, list) :
            # if iterator > recNum: 
            it = (hn.comb_Small_Time[inhibite] + hn.recNum
                if inhibite else hn.comb_Small_Time[inhibite])
            hn.indiNeuronsDetailed[it].append(result)
        # ... and converted back to a float value
            result = result[1] - hn.thresh[iterator]

        ### Record Activation ###
        if result >= 0:
            # Calculate The time it was recorded
            temp = hn.comb_Big_Time[0]+hn.comb_Small_Time[0]/hn.sizeM[0]
            hn.activeOT[iterator].append(temp)
            if not justActive:
                hn.fireOT[iterator].append(temp)

        hn.comb_Small_Time[inhibite] +=1

        ### End of comb_Small_Time Sequence
        if hn.comb_Small_Time[inhibite] >= hn.sizeM[inhibite]:
            # Reset Inner Loop
            hn.comb_Small_Time[inhibite] = 0
            # Tick Up Outer Loop
            hn.comb_Big_Time[inhibite] +=1

            ### Update the sequence of Updates ###
            def update_sequence(hn):
                if hn.randomProcess: 
                    hn.combSequence[inhibite]  = np.random.randint(
                        hn.combMinSize[inhibite],hn.combMaxSize[inhibite], hn.sizeM[inhibite])
                else:
                    hn.combSequence[inhibite] = np.random.permutation(hn.combRange[inhibite])
                return hn
            update_sequence(hn)
            return inhibite

def run_exp(jCon, thresh, external, info, toDo):
    hn = setup(jCon, thresh, external, info, toDo)
    while hn.comb_Big_Time[0] < hn.maxTime:
        inhibite = inside_loop(hn)
        ### Checks whether to update external input function ###
        # 1. Did label change compared to last run, 2. is Switch Active,
        # 3. Is this an excitatory process? 4. Is this the last run already?
        if (hn.switch and inhibite == 0 and hn.comb_Big_Time[0] < hn.maxTime and
            not (hn.labels[hn.comb_Big_Time[0]] == hn.labels[hn.comb_Big_Time[0]-1])):  
            # Update External Input
            external = createExt(info, hn.meanExt_M[hn.labels[hn.comb_Big_Time[0]]])

        ### Print ### 
        # if comb_Big_Time[0] % 10 == 0 and not inhibite and print_GLOBAL:
        #     print(f"{(comb_Big_Time[0]/maxTime):.0%}", end=", ", flush=True)
        #     # if GUI:
        if hn.comb_Big_Time[0] >= hn.print_count and not inhibite and print_GLOBAL:
            while hn.comb_Big_Time[0] >= hn.print_count:
                print(int((hn.print_count/hn.maxTime)*100),end='%, ', flush=True)
                hn.print_count+=hn.print_steps*hn.maxTime
            # print(f"{(comb_Big_Time[0]/maxTime):.0%}", end=", ", flush=True)

    print("")
    return  hn.indiNeuronsDetailed, hn.activeOT, hn.fireOT, hn.labels