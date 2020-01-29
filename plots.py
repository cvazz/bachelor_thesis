from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

import scipy
from scipy import special
from scipy.stats import gaussian_kde
import scipy.integrate as integrate
import scipy.optimize as optimize

from collections import OrderedDict #grouped Labels
import inspect

import time

import utils
#import neuron as nn

###############################################################################
############################### Global Variables ##############################
###############################################################################
savefig_GLOBAL    = 1
showPlots_GLOBAL  = 0
figfolder_GLOBAL  = ""
titletxt_GLOBAL   = ""
captiontxt_GLOBAL = ""
print_GLOBAL = 1
###############################################################################
############################## Utility Functions ##############################
###############################################################################


def finishplot(title_of_plot, xlabel_of_plot, ylabel_of_plot,
               name_of_plot, fig, ignoreY=0):
    plt.title(title_of_plot)
    plt.xlabel(xlabel_of_plot)
    if not ignoreY: 
        plt.ylabel(ylabel_of_plot)
    folder = utils.checkFolder(figfolder_GLOBAL)
    # fullname = utils.testTheName(folder +name_of_plot+titletxt_GLOBAL, "png")
    fullname = utils.testTheName(folder + name_of_plot, "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        if print_GLOBAL:
            utils.plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(fig)


def regr(func, x, y):
    #extracts default values from function
    signature = inspect.signature(func)
    defaults =  [ v.default for k, v in signature.parameters.items()
                if v.default is not inspect.Parameter.empty]
    #actual fit
    popt, pcov = scipy.optimize.curve_fit(func, x, y)#, p0=defaults)
    #varr -> st_dev
    perr = np.sqrt(np.diag(pcov))
    #label
    fit_label =  "Curve Fit: $"+(str(round(popt[0],3))+r'\pm'+str(round(perr[0],3)) 
            + r"\cdot e^{" + str(round(popt[1],3))+r'\pm'+str(round(perr[1],3))+r"}$")
    #y values for plot
    y_pred = func(x,*popt)
    plt.plot(x,y_pred,label=fit_label)
    plt.legend()
###############################################################################
############################## Plotting Functions #############################
###############################################################################


def indiExtended(indiNeuronsDetailed, threshM, recNum):
    captiontxt = captiontxt_GLOBAL
    showRange = recNum
    exORin = 0
    level = 0
    """
    fig, axarr  = plt.subplots(2,sharex=True,)
    ax1         = axarr[0]
    ax2         = axarr[1]
    """
    fig = plt.figure(constrained_layout=False, )#figsize = (10,10))
    h_ratio = 10-showRange/2 if 10-showRange/2>2 else 2
    gs  = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[h_ratio,1])
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
    col = ['blue', 'green', 'red', 'purple', 'grey']
    labelNames = ["positive", "total sum", "negative", "positive inside", "positive external"]
    lstyle = ['-', '-', '-', ':', '--']
    for i in range(level, showRange + level):
        rec = np.transpose(indiNeuronsDetailed[i])
        xs = range(0,len(rec[1]))
        lines=[[] for x in range(showRange)]
        spike = [1 if rec[1][j]
         > threshM[exORin] 
         else 0 for j in range(len(rec[1]))]
        spike += [0.3 for x in range(lengthOfPlot-len(rec[1]))]
        dataspike.append(spike)
        for j in range(len(rec)):
            lines[i].append(ax1.plot(xs,rec[j], color = col[j],
                label= labelNames[j], linestyle=lstyle[j], linewidth = .8))
    ax2.imshow(dataspike, aspect='auto', cmap='Greys', interpolation='nearest')

    xs = range(lengthOfPlot)
    consta = [threshM[exORin] for x in range(lengthOfPlot)]
    ax1.plot(xs, consta, color="black", linestyle="-", linewidth=1.4)
    fig.suptitle('Individual Neuron Firing Pattern', fontsize= 20)
    labelX = "Time"
    # plt.xlabel(labelX + '\n\n' + captiontxt)
    plt.xlabel(labelX + '\n\n' + captiontxt_GLOBAL)
    ax1.set(ylabel='Current')
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    ax2.set(ylabel='Spike')

    folder = utils.checkFolder(figfolder_GLOBAL)
    name = "IndiExt"
    # fullname = utils.testTheName(folder +name+titletxt_GLOBAL , "png")
    fullname = utils.testTheName(folder + name , "png")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        utils.plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(fig)


def newDistri(inputOT, timer):
    actiRowLen = np.array([len(row) for row in inputOT])
    norm       = actiRowLen/np.mean(actiRowLen) 

    uniq = len(np.unique(norm))
    binsize = 10 if uniq <10 else uniq if uniq <timer else timer

    fig = plt.figure(tight_layout=True)
    plt.hist(norm, bins=binsize, weights=np.ones(len(norm))/len(norm))

    disclaimer = "actual amount of dead nodes:" +str(len([x for x in inputOT if not x])/len(inputOT))
    title_of_plot   = 'Firing Rate Distribution' 
    xlabel_of_plot  = 'Normalized Firing Rate'+ '\n\n'+ captiontxt_GLOBAL + "\n" + disclaimer
    ylabel_of_plot  = 'Density'
    name_of_plot    = "Old_Bins_Distri"
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)

    d = np.diff(np.unique(norm)).min()
    first_bin = norm.min() - float(d)/2
    last_bin = norm.max() + float(d)/2
    bin_range = np.arange(first_bin, last_bin + d, d)

    normal_weights = np.ones(len(norm))/len(norm)

    fig = plt.figure(tight_layout = True)
    plt.hist(norm, bins = bin_range, weights = normal_weights)

    disclaimer = "actual amount of dead nodes:" +str(len([x for x in inputOT if not x])/len(inputOT))
    title_of_plot   = 'Firing Rate Distribution' 
    xlabel_of_plot  = 'Normalized Firing Rate'+ '\n\n'+ captiontxt_GLOBAL + "\n" + disclaimer
    ylabel_of_plot  = 'Density'
    name_of_plot    = "Distri"
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)

def newMeanOT(mean_inputOT, isFire=0):
    colors = ['b','r']
    labels = ['Excitatory','Inhibitory']
    fig = plt.figure(tight_layout = True)

    for i in range(2):
        plt.plot(mean_inputOT[i],label=labels[i], color= colors[i])
    plt.legend()
    if isFire   :  title_of_plot   = 'Mean Rate of Fire over Time' 
    else        :  title_of_plot   = 'Mean Rate of Activation over Time' 
    if isFire   :  ylabel_of_plot  = 'Activation Rate'
    else        :  ylabel_of_plot  = 'Firing Rate'
    xlabel_of_plot  = 'Time'+ '\n\n'+ captiontxt_GLOBAL
    name_of_plot    = "meanOT"
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)
     
def fit_func(t,a,b):
    return a*np.exp(b*t)

def newInterspike(inputOT,timer,display_Log = 1):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c
    def exponential(x,a=1,k=1):
        return a*np.exp(x*k)
    # matplotlib.rcParams['text.usetex'] = True
    inpSTART  = [np.array(row[:-1]) for row in inputOT]
    inpEND    = [np.array(row[1:])  for row in inputOT]
    inpDIFF   = [inpEND[i] - inpSTART[i] for i in range(len(inputOT))]
    flat_diff = np.array([x for row in inpDIFF for x in row])

    fig = plt.figure(tight_layout = True)
    bin_range = np.arange(min(flat_diff), max(flat_diff),0.5)
    weights_ = np.ones(len(flat_diff))/len(flat_diff)
    n, binz, _ = plt.hist(flat_diff, bins = bin_range, weights = weights_)
    if display_Log:
        plt.yscale('log', nonposy='clip')
    bincenters = 0.5*(binz[1:]+binz[:-1])
    try:
        regr(func,bincenters,n)
    except:
        print("Fit Failed")
    # popt, pcov = scipy.optimize.curve_fit(exponential, bincenters, n, p0=[1,-0.5])
    # perr = np.sqrt(np.diag(pcov))
    # fit_label =  "Curve Fit: $"+(str(round(popt[0],3))+r'\pm'+str(round(perr[0],3)) 
    #         + r"\cdot e^{" + str(round(popt[1],3))+r'\pm'+str(round(perr[1],3))+r"}$")
    # y_pred = exponential(bincenters,*popt)
    # plt.plot(bincenters,y_pred,label=fit_label)
    # plt.legend()

    # captiontxt = captiontxt_GLOBAL.replace('_','\_')
    name_of_plot    = "interspike" 
    title_of_plot   = 'Interspike Interval' 
    xlabel_of_plot  = 'Time'+ '\n\n'+ captiontxt_GLOBAL
    ylabel_of_plot  = 'Density'
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)

def dots2(inputOT, timer):
    precision = 1#.1
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    val_OT = np.zeros((len(inputOT),int(timer/precision)))
    for i in range(len(inputOT)):
        for zeit in inputOT[i]:
            val_OT[i,int(zeit/precision)] = 1 
    ax.imshow(val_OT, aspect='auto', cmap='Greys',extent = [0,timer,0,len(inputOT)], interpolation='nearest')

    # mrkr = math.sqrt(1/(timer*len(inputOT)))*300
    # for i,row in enumerate(inputOT):
    #     yaxis = [i for _ in row]
    #     plt.plot(row,yaxis, marker= 's', color= 'black', markersize= mrkr,linestyle='none',)
    ### Kein Rand ###
    ax.set_ylim(ymin=0,ymax=len(inputOT))
    ax.set_xlim(xmin=0,xmax= timer)
    ### Keine Ticks
    ax.set_yticks([])
    ### Background Colour und Beschriftung ###
    plt.text(0.06, 0.7, "Inhibitory", fontsize=8, rotation=90,
        transform=plt.gcf().transFigure)
    plt.text(0.06, 0.36, "Excitatory", fontsize=8, rotation=90,
        transform=plt.gcf().transFigure)
    plt.axhspan(0, len(inputOT)/2, facecolor='red', alpha=0.3)
    plt.axhspan(len(inputOT)/2,len(inputOT), facecolor='blue', alpha=0.3)

    title_of_plot   = 'Neuronal Activity over Time' 
    xlabel_of_plot  = 'Time'+ '\n\n'+ captiontxt_GLOBAL
    ylabel_of_plot  = 'Neurons\n'
    name_of_plot    = "dots2"
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)

def mean_vs_ext(meanList, isFire = 0):
    fig = plt.figure(tight_layout = True)
    ax = fig.add_subplot(111)
    pltColor = ['b','r']
    pltLabel = ['Excitatory','Inhibitory']
    lstyle   = ['-','--']
    for i in range(2):
        plt.plot(meanList[0],meanList[i+1],color=pltColor[i],
                    label=pltLabel[i],linestyle=lstyle[i], marker='o')
    plt.legend()
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    title_of_plot   = '' 
    if isFire   :   ylabel_of_plot  = 'Mean of Firing Rate'
    else        :   ylabel_of_plot  = 'Mean of Activation Rate'
    xlabel_of_plot  = 'External Rate'
    name_of_plot    = "mean_vs_ext"
    finishplot(title_of_plot , xlabel_of_plot, ylabel_of_plot, name_of_plot,fig)

