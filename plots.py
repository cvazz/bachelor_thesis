
from pathlib import Path
import numpy as np
from scipy import special
import scipy.integrate as integrate

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from collections import OrderedDict #grouped Labels

import utils

###############################################################################
############################### Global Variables ##############################
###############################################################################
savefig_GLOBAL = 1
showPlots_GLOBAL = 0


def mean_distri(figfolder, total_times_one, fireCount, timer, titletxt, captiontxt):
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

    folder = utils.checkFolder(figfolder)
    name = "density"
    fullname = utils.testTheName(folder +name+titletxt , "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        utils.plotMessage(fullname)
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

    folder = utils.checkFolder(figfolder)
    name = "histogram"
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

def interspike(figfolder, nval_over_time, timer, titletxt, captiontxt,display_Log = 1):
    """
    see analyze for calculation
    """
    
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
    xlabel = "time"

    plt.hist(diff, bins = binsize, weights = np.ones(len(diff))/len(diff))
    plt.title('Interspike Interval')
    plt.xlabel(xlabel+'\n\n' + captiontxt)
    plt.ylabel('density')
    if display_Log:
        plt.yscale('log', nonposy='clip')


    folder = utils.checkFolder(figfolder)
    name = "intervalHist"
    fullname = utils.testTheName(folder +name+titletxt , "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        utils.plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(histfig)

def dots(figfolder, nval_over_time, timer, titletxt, captiontxt):

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


    folder = utils.checkFolder(figfolder)
    name = "dots"
    fullname = utils.testTheName(folder +name+titletxt , "png")
    # ax.annotate('test', xy=(0.0, 0.75), xytext=(-0.1, 0.750), xycoords='axes fraction', 
    #         fontsize=12, ha='center', va='bottom',
    #         bbox=dict(boxstyle='square', fc='white'),
    #         arrowprops=dict(arrowstyle='-[, widthB=7.0, lengthB=1.5', lw=2.0))
    plt.axhspan(0, len(nval_over_time)/2, facecolor='blue', alpha=0.3)
    plt.axhspan(len(nval_over_time)/2,len(nval_over_time), facecolor='red', alpha=0.3)
    if savefig_GLOBAL:
        plt.savefig(fullname)
        utils.plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(fig)