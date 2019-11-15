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

################################################################################
############################## Global Variables ################################
################################################################################
savefig_GLOBAL = 1
showPlots_GLOBAL = 0
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

def mostRecent(vfolder):
    """
    Finds last added "testreihe" in given folder
    
    :param vfolder: folder to look for testreihe
    :type vfolder: class: 'pathlib.PosixPath'
    :return: name of "max" testreihe
    :rtype: string
    """
    paths = list(vfolder.glob('**'))
    names =  [path.name for path in paths]
    filtered = [name for name in names if "testreihe" in name]
    return max(filtered)

def testthename(name,fileEnding):
    workinName = name + "." + fileEnding
    pathname = Path(workinName)
    count = 0
    while pathname.exists():
        count += 1
        workinName = name +"_no_" +str(count) + "." + fileEnding
        pathname = Path(workinName)
    return workinName

def makeNewPath(valueFolder, name, fileEnding):
    workinName = name + "." + fileEnding
    pathname = Path(valueFolder / workinName)
    count = 0
    while pathname.exists():
        count += 1
        workinName = name +"_no_" +str(count) + "." + fileEnding
        pathname = Path(valueFolder / workinName)
    return pathname

def makeExistingPath(valueFolder, name, fileEnding):
    workinName = name + "." + fileEnding
    pathname = Path(valueFolder / workinName)
    if pathname.exists():
        return pathname
    else: 
        raise NameError("no file with this name exists")

def checkFolder(figfolder):
    folderpath = Path(figfolder)
    if not folderpath.exists():
        folderpath.mkdir()
    return figfolder + "/"

def relMax(fireCount,showRange):
    return np.argpartition(fireCount, -1*showRange)[-1*showRange:]

def rowSums(matrix):
    total = []
    for row in matrix:
        total.append(sum(row))
    return total

def plotMessage(fullname):
    if plotLoc_print2console_GLOBAL:
        print("plotted and saved at: " + fullname)

def timeOut(timediff):
    printstring = ""
    if timediff>200:
        mins = int(timediff/60)
        secs = timediff-mins*60
        if mins> 100:
            hours = int (mins/60)
            mins  = mins%60
            printstring += f'{hours}h, '
        printstring += f'{mins}m and '
    else: 
        secs = timediff
    printstring += f'{secs:3.4}s'
    print(printstring)

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

    indiName        = makeNewPath(valueFolder, indiNametxt, "npy")
    fireName        = makeNewPath(valueFolder, fireNametxt, "npy")
    #nvalOTName      = makeNewPath(valueFolder, nvalOTNametxt, "npy")
    nval_Index_Name = makeNewPath(valueFolder, nvalIndexNametxt, "npy")
    infoName        = makeNewPath(valueFolder, infoNametxt, "pkl")

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

    indiName        = makeExistingPath(valueFolder, indiNametxt, "npy")
    fireName        = makeExistingPath(valueFolder, fireNametxt, "npy")
    #nvalOTName      = makeExistingPath(valueFolder, nvalOTNametxt, "npy")
    nval_Index_Name = makeExistingPath(valueFolder, nvalIndexNametxt, "npy")
    infoName        = makeExistingPath(valueFolder, infoNametxt, "pkl")

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
############################# Plotting Functions ###############################
################################################################################

def plotTotal(figfolder, total_times_one, fireCount, timer, titletxt, captiontxt):
    """
    plots distribution of firing pattern in relation to mean firing pattern

    (currently everything larger than 5*mean is labelled as 6)

    :param total_times_one: contains all the times a specific neuron fired
    :param fireCount: contains all the times a specific neuron spiked
        (ie turned 0 afterwards) (not in use)
    """
    total_times_one = np.array(total_times_one)
    meanTot = np.mean(total_times_one)
    if meanTot == 0:
        print("not a single flip for the following starting values")
    total_times_one = total_times_one/meanTot
    density = gaussian_kde(total_times_one)
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

    folder = checkFolder(figfolder)
    name = "density"
    fullname = testthename(folder +name+titletxt , "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        plotMessage(fullname)
    #plt.show()
    plt.close(fig)
    
    histfig = plt.figure()
    uniq = len(np.unique(total_times_one))
    binsize = 10 if uniq <10 else uniq if uniq<timer else timer
    plt.hist(total_times_one, bins = binsize, weights = np.ones(len(total_times_one))/len(total_times_one))
    plt.title('Fire Rate Distribution')
    plt.xlabel('fireCount rate/mean')
    plt.ylabel('density')
    histfig.text(.5,.05,captiontxt, ha='center')
    histfig.subplots_adjust(bottom=0.2)

    folder = checkFolder(figfolder)
    name = "histogram"
    fullname = testthename(folder +name+titletxt , "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(histfig)

def plotIndi(figfolder, indiNeuronsDetailed, fireCount, threshM, titletxt, captiontxt):
    """
    Plots inputs in several neurons (ie 3 to 10)

    Shows positive, negative and total_times_one input for several neurons 

    :param indiNeuronsDetailed: Contains pos, neg, and total_times_one value 
                                for a subgroup of neurons at each timestep

    """
    showRange = 5
    exORin = 0
    fig = plt.figure()
    recSize = len(indiNeuronsDetailed)
    showTheseRows = relMax(fireCount[:recSize],showRange)
    for i in showTheseRows:
        rec = np.array(indiNeuronsDetailed[i])
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

    folder = checkFolder(figfolder)
    name = "Indi"
    fullname = testthename(folder +name+titletxt , "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(fig)


def plotIndiExtended(figfolder, indiNeuronsDetailed, fireCount, threshM,
    showRange, titletxt, captiontxt):
    showRange = 15
    exORin = 0
    level = 0
    """
    fig, axarr  = plt.subplots(2,sharex=True,)
    ax1         = axarr[0]
    ax2         = axarr[1]
    """
    fig = plt.figure(constrained_layout = False, figsize = (10,10))
    h_ratio = 10-showRange/2 if 10-showRange/2>2 else 2
    gs  = fig.add_gridspec(ncols=1,nrows=2,height_ratios=[h_ratio,1])
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,:], sharex =ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_yticks([])
    dataspike = []
    lengthOfLists =  [len(row) for row in indiNeuronsDetailed]
    minMaxDiff = np.max(lengthOfLists) - np.min(lengthOfLists)
    if minMaxDiff:
        print("WARNING: Asymmetric Growth of individual neurons recorded")
        captiontxt += (f"\n  unequal size of neurons, difference between max" +
                f" and min = {minMaxDiff}")
    lengthOfPlot = max(lengthOfLists)
    for i in range(level, showRange + level):
        rec = np.transpose(indiNeuronsDetailed[i])
        xs = range(0,len(rec[1]))
        col=['blue', 'green', 'red']
        labelNames = ["positive", "total sum", "negative"]
        lines=[[] for x in range(showRange)]
        spike = [1 if rec[1][j] > threshM[exORin] else 0 for j in range(len(rec[1]))]
        spike += [0.3 for x in range(lengthOfPlot-len(rec[1]))]
        dataspike.append(spike)
        for j in range(len(rec)):
            lines[i].append(ax1.plot(xs,rec[j], color = col[j],
                label= labelNames[j], linewidth = .8))
    ax2.imshow(dataspike, aspect='auto', cmap='Greys', interpolation='nearest')

    xs = range(lengthOfPlot)
    consta = [threshM[exORin] for x in range(lengthOfPlot)]
    ax1.plot(xs,consta, color = "black", linewidth = 2.0)
    #fig.text(.5,.05,captiontxt, ha='center')
    #fig.subplots_adjust(bottom=0.3)
    fig.suptitle('Individual Neuron Firing Pattern', fontsize= 20)
    labelX = "time"
    plt.xlabel(labelX + '\n\n' + captiontxt)
    ax1.set(ylabel = 'Current')
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    ax2.set(ylabel = 'Spike')
    #Labels
    #first_legend = plt.legend(handles=[lines[0]], loc='lower right')
    #ax1 = plt.gca().add_artist(first_legend)
    #aplt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',   
           #ncol=2, mode="expand", borderaxespad=0.)


    folder = checkFolder(figfolder)
    name = "IndiExt"
    fullname = testthename(folder +name+titletxt , "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(fig)


def analyzeTau(rec):
    """
    Calculates Intervals between Firing
    
    Example: Calculates fire rate of [0,0,0,1,0,1,0,0,1,1,1,1,1,1,0,1]
    as [1, 2, 1]

    :param rec: 
    :type rec: list (binary)
    """
    dist =[]
    buff = 1
    counter = 0
    if not type(rec) == list:
        rec = rec.tolist()
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

def plotInterspike(figfolder, nval_over_time, timer, titletxt, captiontxt):
    """
    see analyze for calculation
    """
    display_Log = 0
    diff = []
    for rec in nval_over_time:
        a = analyzeTau(rec.tolist())
        if a: 
            diff += a

    extratxt = ""
    histfig = plt.figure(tight_layout = True)
    uniq = len(np.unique(diff))
    binsize = 10 if uniq <10 else uniq if uniq<timer else timer
    captiontxt += extratxt

    plt.hist(diff, bins = binsize, weights = np.ones(len(diff))/len(diff))
    plt.title('Interspike Interval')
    plt.xlabel('time\n\n' + captiontxt)
    plt.ylabel('density')
    if display_Log:
        plt.yscale('log', nonposy='clip')


    folder = checkFolder(figfolder)
    name = "intervalHist"
    fullname = testthename(folder +name+titletxt , "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(histfig)

def plotDots(figfolder, nval_over_time, timer, titletxt, captiontxt):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    record =np.transpose(nval_over_time)
    #ax.imshow(record, aspect='auto', cmap='Greys', interpolation='nearest')
    ax.imshow(nval_over_time, aspect='auto', cmap='Greys', interpolation='nearest')
    plt.title('Neurons firing over time')
    plt.xlabel('time')
    plt.ylabel('neurons\n')
    ax.set_yticks([])
    plt.text(0.1, 0.65, "excitatory", fontsize=8, rotation=90,
        transform=plt.gcf().transFigure)
    plt.text(0.1, 0.33, "inhibitory", fontsize=8, rotation=90,
        transform=plt.gcf().transFigure)
    ax.set_ylim(ymax=0)
    ax.set_xlim(xmin=0)
    fig.text(.5,.05,captiontxt, ha='center')
    fig.subplots_adjust(bottom=0.2)


    folder = checkFolder(figfolder)
    name = "dots"
    fullname = testthename(folder +name+titletxt , "png")
    # ax.annotate('test', xy=(0.0, 0.75), xytext=(-0.1, 0.750), xycoords='axes fraction', 
    #         fontsize=12, ha='center', va='bottom',
    #         bbox=dict(boxstyle='square', fc='white'),
    #         arrowprops=dict(arrowstyle='-[, widthB=7.0, lengthB=1.5', lw=2.0))
    plt.axhspan(0, len(nval_over_time)/2, facecolor='blue', alpha=0.3)
    plt.axhspan(len(nval_over_time)/2,len(nval_over_time), facecolor='red', alpha=0.3)
    if savefig_GLOBAL:
        plt.savefig(fullname)
        plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(fig)
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
    timeOut(np.mean(np.array(timediff)))
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
    timeOut(timeend - timestart)
    return sizeM, threshM, extM, jCon, thresh, external





def testRoutine(
    timer, K, mean0, tau,
    sizeM,threshM, extM, recNum,
    jCon, thresh, external, figfolder, valueFolder,
    jE, jI, titletxt = "", captiontxt = "",
    doSequ = 1, doPoiss = 0, doRand = 0,
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
    timeOut(timeend - timestart)
    
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
        plotTotal(figfolder, total_times_one,fireCount, timer, titletxt, captiontxt)
    if pIndi:
        plotIndi(figfolder,indiNeuronsDetailed,fireCount, threshM, titletxt, captiontxt)
    if pIndiExt:
        #print(np.sum(indiNeuronsDetailed[1]))
        plotIndiExtended(figfolder,indiNeuronsDetailed,fireCount, threshM, recNum, titletxt, captiontxt)
    if pInterspike:
        plotInterspike(figfolder, nval_over_time, timer, titletxt, captiontxt)
    if pDots:
        #print(sum(nval_over_time))
        plotDots(figfolder, nval_over_time, timer, titletxt, captiontxt)
    


    del indiNeuronsDetailed
    del fireCount
    del nval_over_time
    del total_times_one   
    return meanOT


def afterSimulationAnalysis():
    useMostRecent = 1
    vfolder = Path("ValueVault")
    if useMostRecent:
        loadTimeStr = mostRecent(vfolder)
    else:
        loadTimeStr = "testreihe_191111_1038" #create a "find most recent" function
    valueFolder =  vfolder / loadTimeStr

    timestr = time.strftime("%y%m%d_%H%M")
    figfolder = "figs/testreihe_" + timestr

    showRange = 15
    indiNeuronsDetailed, fireCount, nval_over_time, infoDict = recoverResults(valueFolder)
    total_times_one = rowSums((nval_over_time))
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
        plotTotal(figfolder, total_times_one,fireCount, timer, titletxt, captiontxt)
    if pIndi:
        plotIndi(figfolder,indiNeuronsDetailed,fireCount, threshM, titletxt, captiontxt)
    if pIndiExt:
        plotIndiExtended(figfolder,indiNeuronsDetailed,fireCount, threshM, showRange, titletxt, captiontxt)
    if pInterspike:
        plotInterspike(figfolder, nval_over_time, timer, titletxt, captiontxt)
    if pDots:
        plotDots(figfolder, nval_over_time, timer, titletxt, captiontxt)

def parameters():
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
    timer   = 300
    K       = 1000
    size    = 10000
    sizeE   = size
    sizeI   = size

    return timer, sizeE, sizeI, extE, extI, jE, jI, threshE, threshI, tau, mean0, K 

def main():
    timestr = time.strftime("%y%m%d_%H%M")
    figfolder = "figs/testreihe_" + timestr
    valuefoldername = "ValueVault/testreihe_"
    valueFolder =  Path(valuefoldername + timestr)
    timer, sizeE, sizeI, extE, extI, jE, jI, threshE, threshI, tau, mean0, K = parameters()

    #backup_correctparameters()
    #troubleshootParamters()

    #Bools for if should be plotted or not
    pTot        = 1
    pIndiExt    = 1
    pInterspike = 1
    pDots       = 1
    pIndi       = 0
    #Recording Specification(should be reviewed):
    recNum      = 100

    #Only one Sequence per Routine
    doSequ      = 0
    doPoiss     = 1 
    doRand      = 0 
    
    print("Ergebnisse abspeichern und danach plotten")
    print("What exactly is Poisson statistics and process")
    extratxt = ""

    sizeM, threshM, extM, jCon, thresh, external = prepare(
        K, mean0, tau, sizeE, sizeI, extE, extI, jE, jI,
        threshE, threshI)

    sizeMax = np.sum(sizeM)
    #CURRENTLY BEING OVERWRITTEN
    titletxt = f'_S_{str(sizeMax)[:1]}e{int(np.log10(sizeMax))}_K_{(K)}'#_m0_{str(mean0)[2:]}'# \njE: {jE}   jI: {jI} '
    captiontxt = f'Network Size: {sizeMax}  K: {K}  mean_0: {mean0} \n\
        time: {timer}   jE: {jE}   jI: {jI} ' + extratxt

    # mean100 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
    #         0.09, 0.1  , 0.125, 0.15 , 0.175, 0.2  , 0.225, 0.25 , 0.275] 
    # mean100 = [0.01, 0.04]# 0.1, 0.2]

    # meanOT = []
    # for i,mean0_itt in enumerate(mean100):
        # meanOT.append(
    testRoutine(
        timer, K, mean0, tau,
        sizeM, threshM, extM, recNum,
        jCon, thresh, external, figfolder, valueFolder,
        jE, jI,titletxt, captiontxt,
        doSequ, doPoiss, doRand,
        pTot, pIndi, pIndiExt, pInterspike,pDots)
        # print(meanOT)
    doSequ      = 0
    doPoiss     = 0 
    doRand      = 1 
    
    testRoutine(
        timer, K, mean0, tau,
        sizeM, threshM, extM, recNum,
        jCon, thresh, external, figfolder, valueFolder,
        jE, jI,titletxt, captiontxt,
        doSequ, doPoiss, doRand,
        pTot, pIndi, pIndiExt, pInterspike,pDots)
    # sizeM, threshM, extM, jCon, thresh, external = prepare(
    #     K, mean0, tau, sizeE, sizeI, extE, extI, jE, jI,
    #     threshE, threshI)
    # mean100+=meanOT
    # mean100 = np.array(mean100)
    # mean100.shape = (2,len(meanOT))
    # print(mean100)
    # np.save("mean0vsmeanOT",mean100)
    
if __name__ == "__main__":
    #afterSimulationAnalysis()
    main()

