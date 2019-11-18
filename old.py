import numpy as np

import neuron
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
    prob =  tau / (1+tau)

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
        if np.random.uniform(0,1)<prob:
            iterator = exSequence[exIter]
            if iterator <recNum:
                vals = neuron.timestepMatRecord(iterator, nval, jCon,
                    thresh, external, fireCount, sizeM)
                indiNeuronsDetailed[iterator].append(vals)
                if vals[1] >= 1:
                    nval_over_time[iterator,exTime] += 1
            else:
                overThresh = neuron.timestepMat (iterator, nval, jCon, thresh, external, fireCount)
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
            overThresh = neuron.timestepMat (iterator, nval, jCon, thresh, external, fireCount)
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
    return [pos,summe + external[iter],neg,thresh[iter]]