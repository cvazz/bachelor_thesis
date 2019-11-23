from pathlib import Path
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import special
import scipy.integrate as integrate
from scipy.stats import gaussian_kde
from collections import OrderedDict #grouped Labels

import time

import utils
import neuron as nn

###############################################################################
############################### Global Variables ##############################
###############################################################################
savefig_GLOBAL      = 0
showPlots_GLOBAL    = 1
figfolder_GLOBAL    = ""
titletxt_GLOBAL     = ""
captiontxt_GLOBAL   = ""

###############################################################################
############################## Plotting Functions #############################
###############################################################################

def finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig):
    plt.title(title_of_plot)
    plt.xlabel(xlabel_of_plot )
    plt.ylabel(ylabel_of_plot)
    folder = utils.checkFolder(figfolder_GLOBAL)
    fullname = utils.testTheName(folder +name_of_plot+titletxt_GLOBAL, "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        utils.plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(fig)

def activation_distribution(figfolder, fireCount, timer, titletxt, captiontxt):
    """
    plots distribution of firing pattern in relation to mean firing pattern

    (currently everything larger than 5*mean is labelled as 6)

    :param total_times_one: contains all the times a specific neuron fired
    :param fireCount: contains all the times a specific neuron spiked
        (ie turned 0 afterwards) (not in use)
    """
    total_times_one = np.array(fireCount)
    meanTot = np.mean(total_times_one)
    # total_times_one = total_times_one/meanTot
    # density = gaussian_kde(total_times_one)
    # xs = np.linspace(0,3)
    # density.covariance_factor = lambda : .1
    # density._compute_covariance()
    # fig = plt.figure()
    # plt.plot(xs,density(xs))
    
    histfig = plt.figure(tight_layout = True)
    uniq = len(np.unique(total_times_one))
    binsize = 10 if uniq <10 else uniq if uniq<timer else timer
    normal_tto = total_times_one/np.mean(total_times_one)
    plt.hist(total_times_one, bins = binsize, 
        weights = np.ones(len(total_times_one))/len(total_times_one))
    plt.title('Fire Rate Distribution')
    plt.xlabel('fireCount rate/mean\n\n'+captiontxt)
    plt.ylabel('density')

    folder = utils.checkFolder(figfolder)
    name = "ActiDist"
    fullname = utils.testTheName(folder +name+titletxt , "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        utils.plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(histfig)

def indi(figfolder, indiNeuronsDetailed, fireCount, threshM, titletxt, captiontxt):
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
    showTheseRows = utils.relMax(fireCount[:recSize],showRange)
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

    folder = utils.checkFolder(figfolder)
    name = "Indi"
    fullname = utils.testTheName(folder +name+titletxt , "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        utils.plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(fig)


def indiExtended(figfolder, indiNeuronsDetailed, fireCount, threshM,
    titletxt, captiontxt,recNum):

    showRange = recNum
    exORin = 0
    level = 0
    """
    fig, axarr  = plt.subplots(2,sharex=True,)
    ax1         = axarr[0]
    ax2         = axarr[1]
    """
    fig = plt.figure(constrained_layout = False, )#figsize = (10,10))
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
        spike = [1 if rec[1][j]
         > threshM[exORin] 
         else 0 for j in range(len(rec[1]))]
        spike += [0.3 for x in range(lengthOfPlot-len(rec[1]))]
        dataspike.append(spike)
        for j in range(len(rec)):
            lines[i].append(ax1.plot(xs,rec[j], color = col[j],
                label= labelNames[j], linewidth = .8))
    ax2.imshow(dataspike, aspect='auto', cmap='Greys', interpolation='nearest')

    xs = range(lengthOfPlot)
    consta = [threshM[exORin] for x in range(lengthOfPlot)]
    ax1.plot(xs,consta, color = "black", linewidth = 2.0)
    fig.suptitle('Individual Neuron Firing Pattern', fontsize= 20)
    labelX = "time"
    plt.xlabel(labelX + '\n\n' + captiontxt)
    ax1.set(ylabel = 'Current')
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    ax2.set(ylabel = 'Spike')

    folder = utils.checkFolder(figfolder)
    name = "IndiExt"
    fullname = utils.testTheName(folder +name+titletxt , "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        utils.plotMessage(fullname)
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

def interspike(figfolder, nval_over_time, timer, display_Log = 1):
    """
    see analyze for calculation
    """
    
    diff = []
    for rec in nval_over_time:
        a = analyzeTau(rec.tolist())
        if a: 
            diff += a

    fig = plt.figure(tight_layout = True)
    uniq = len(np.unique(diff))
    binsize = 10 if uniq <10 else uniq if uniq<timer else timer

    plt.hist(diff, bins = binsize, weights = np.ones(len(diff))/len(diff))
    if display_Log:
        plt.yscale('log', nonposy='clip')

    name_of_plot    = "intervalHist"
    title_of_plot   = 'Interspike Interval' 
    xlabel_of_plot  = 'Time'+ '\n\n'+ captiontxt_GLOBAL
    ylabel_of_plot  = 'Density'
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)

def dots(figfolder, nval_over_time, timer):

    fig = plt.figure(tight_layout = True)
    ax = fig.add_subplot(111)
    record =np.transpose(nval_over_time)
    #ax.imshow(record, aspect='auto', cmap='Greys', interpolation='nearest')
    ax.imshow(nval_over_time, aspect='auto', cmap='Greys', interpolation='nearest')

    ### Kein Rand ###
    ax.set_ylim(ymax=0)
    ax.set_xlim(xmin=0)
    ### Keine Ticks
    ax.set_yticks([])

    ### Background Colour und Beschriftung ###
    plt.text(0.06, 0.7, "excitatory", fontsize=8, rotation=90,
        transform=plt.gcf().transFigure)
    plt.text(0.06, 0.36, "inhibitory", fontsize=8, rotation=90,
        transform=plt.gcf().transFigure)
    plt.axhspan(0, len(nval_over_time)/2, facecolor='blue', alpha=0.3)
    plt.axhspan(len(nval_over_time)/2,len(nval_over_time), facecolor='red', alpha=0.3)

    title_of_plot   = 'Neurons firing over time' 
    xlabel_of_plot  = 'time'+ '\n\n'+ captiontxt_GLOBAL
    ylabel_of_plot  = 'neurons\n'
    name_of_plot    = "dots"
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)

def meanOT(figfolder, nval_over_time, sizeM, timer):
    activationE = []
    activationI = []
    for vals in np.transpose(nval_over_time):
        activationE.append(np.mean(vals[:sizeM[0]]))
        activationI.append(np.mean(vals[sizeM[0]:]))
    fig = plt.figure(tight_layout = True)

    xside = np.arange(len(activationE))
    lineE = plt.plot(xside, activationE, color = "blue",label = "excitatory")
    xside = np.arange(len(activationI))
    lineE = plt.plot(xside, activationI, color = "red", label = "inhibitory")

    title_of_plot   = 'Mean Activation over Time' 
    xlabel_of_plot  = 'time'+ '\n\n'+ captiontxt_GLOBAL
    ylabel_of_plot  = 'Activation Rate'
    name_of_plot    = "old_meanOT"
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)




def newDistri( inputOT, timer ):
    actiRowLen = np.array([len(row) for row in inputOT])
    norm       = actiRowLen/np.mean(actiRowLen) 

    d = np.diff(np.unique(norm)).min()
    first_bin = norm.min() - float(d)/2
    last_bin = norm.max() + float(d)/2
    bin_range = np.arange(first_bin, last_bin + d, d)

    fig = plt.figure(tight_layout = True)
    plt.hist(norm, bins = bin_range, weights = np.ones(len(norm))/len(norm))

    disclaimer = "actual amount of dead nodes:" +str(len([x for x in inputOT if not x])/len(inputOT))
    title_of_plot   = 'Fire Rate Distribution' 
    xlabel_of_plot  = 'fireCount rate/mean'+ '\n\n'+ captiontxt_GLOBAL + "\n" + disclaimer
    ylabel_of_plot  = 'Density'
    name_of_plot    = "Distri"
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)

def newMeanOT(mean_inputOT):
    colors = ['b','r']
    labels = ['excitatory','inhibitory']
    fig = plt.figure(tight_layout = True)

    for i in range(2):
        x_range = np.arange(len(mean_inputOT[i]))
        plt.plot(mean_inputOT[i],label=labels[i], color= colors[i])
    plt.legend()
    title_of_plot   = 'Mean Activation over Time' 
    xlabel_of_plot  = 'time'+ '\n\n'+ captiontxt_GLOBAL
    ylabel_of_plot  = 'Activation Rate'
    name_of_plot    = "better_meanOT"
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)
     

def newInterspike(inputOT,timer,display_Log = 1):
    inpSTART  = [np.array(row[:-1]) for row in inputOT]
    inpEND    = [np.array(row[1:])  for row in inputOT]
    inpDIFF   = [inpEND[i] - inpSTART[i] for i in range(len(inputOT))]
    flat_diff = np.array([x for row in inpDIFF for x in row])

    fig = plt.figure(tight_layout = True)
    uniq = len(np.unique(flat_diff))
    binsize = 10 if uniq <10 else uniq if uniq<timer else timer

    plt.hist(flat_diff, bins = binsize, weights = np.ones(len(flat_diff))/len(flat_diff))
    if display_Log:
        plt.yscale('log', nonposy='clip')

    name_of_plot    = "interspike"
    title_of_plot   = 'Interspike Interval' 
    xlabel_of_plot  = 'Time'+ '\n\n'+ captiontxt_GLOBAL
    ylabel_of_plot  = 'Density'
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)

def dots2(inputOT, timer):
    precision = 1#.1
    mrkr = math.sqrt(1/(timer*len(inputOT)))*300
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    start = time.time()
    val_OT = np.zeros((len(inputOT),int(timer/precision)))
    for i in range(len(inputOT)):
        for zeit in inputOT[i]:
            val_OT[i,int(zeit/precision)] = 1 
    ax.imshow(val_OT, aspect='auto', cmap='Greys',extent = [0,timer,0,len(inputOT)], interpolation='nearest')

    # for i,row in enumerate(inputOT):
    #     yaxis = [i for _ in row]
    #     plt.plot(row,yaxis, marker= 's', color= 'black', markersize= mrkr,linestyle='none',)
    end = time.time()
    # utils.timeOut(end-start)
    ### Kein Rand ###
    ax.set_ylim(ymin=0,ymax=len(inputOT))
    ax.set_xlim(xmin=0,xmax= timer)
    ### Keine Ticks
    ax.set_yticks([])
    ### Background Colour und Beschriftung ###
    plt.text(0.06, 0.7, "inhibitory", fontsize=8, rotation=90,
        transform=plt.gcf().transFigure)
    plt.text(0.06, 0.36, "excitatory", fontsize=8, rotation=90,
        transform=plt.gcf().transFigure)
    plt.axhspan(0, len(inputOT)/2, facecolor='red', alpha=0.3)
    plt.axhspan(len(inputOT)/2,len(inputOT), facecolor='blue', alpha=0.3)

    title_of_plot   = 'Neurons firing over time' 
    xlabel_of_plot  = 'time'+ '\n\n'+ captiontxt_GLOBAL
    ylabel_of_plot  = 'neurons\n'
    name_of_plot    = "dots2"
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)

def mean_vs_ext(meanList):
    fig = plt.figure(tight_layout = True)
    ax = fig.add_subplot(111)
    pltColor = ['b','r']
    pltLabel = ['excitatory','inhibitory']
    for i in range(2):
        plt.scatter(meanList[0],meanList[i+1],color=pltColor[i],label=pltLabel[i])
    plt.legend()
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    title_of_plot   = '' 
    xlabel_of_plot  = 'External Rate'
    ylabel_of_plot  = 'Mean Rate'
    name_of_plot    = "mean_vs_ext"
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)

