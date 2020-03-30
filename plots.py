# from pathlib import Path
import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib.legend import Legend

import scipy
from pathlib import Path
from scipy import special
# from scipy.stats import gaussian_kde
# import scipy.integrate as integrate
import scipy.optimize as optimize

from collections import OrderedDict  # grouped Labels
# import inspect

# import time

import utils
import evaluate as eva
# import neuron as nn

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
    # plt.title(title_of_plot)
    plt.xlabel(xlabel_of_plot)
    if not ignoreY: 
        plt.ylabel(ylabel_of_plot)
    folder = utils.checkFolder(figfolder_GLOBAL)
    # fullname = utils.testTheName(folder +name_of_plot+titletxt_GLOBAL, "pdf")
    fullname = utils.testTheName(folder + name_of_plot, "pdf")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        if print_GLOBAL:
            utils.plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(fig)

def stringy(num):
    if abs(num) < .0001 or num > 100:
        return "%.2E" % num
    else:
        return str(round(num, 4))
def stringy2(num, err):
    expo = abs(num) if abs(num) < err else err
    expo = np.floor(np.log10(expo))
    if expo < -3:
        num = round(num/(10**expo),1)
        err = round(err/(10**expo),1)
        return "(%.0f \\pm %.0f) \\cdot 10^{%d}" % (num, err, expo)
    else:
        return "(%.3f \\pm %.3f) " % (num, err)

first_reg_GLOBAL    = 0

def regr(f, x, y, ax):
    # extracts default values from function
    # signature = inspect.signature(func)
    # defaults = [ v.default for k, v in signature.parameters.items()
    #             if v.default is not inspect.Parameter.empty]
    # actual fit
    if f == "linear":
        def func(x, m, b, ):
            return m * x + b
    elif f == "exp":
        def func(x, b):
            return 1/b * np.exp(-b * x) 
    elif f == "exp2":
        def func(x, a, b, c):
            return a * np.exp(-b * x + c) 
    elif f == "poisson":
        def func(k, lambd):
            return lambd ** k / special.factorial(k) * np.exp(-lambd) 
    else:
        raise NameError("XXXX")
    popt, pcov = scipy.optimize.curve_fit(func, x, y)  # , p0=defaults)
    # varr -> st_dev
    perr = np.sqrt(np.diag(pcov))

    if f == "linear":
        fit_label = (r"Curve Fit: $m \cdot  x+b"
                    + f"$\nm = $" + stringy2(popt[0], perr[0])
                    + f"$, b = $" + stringy2(popt[1], perr[1]) + "$")
    elif f == "exp":
        fit_label = (r"Curve Fit: $ \frac{1}{b}\cdot e^{b x}" 
                    + f"$\nb = $" + stringy2(popt[0], perr[0])
                    + "$")
    elif f == "exp2":
        fit_label = ("Curve Fit: $(" + stringy(popt[0]) + r'\pm' 
                    + stringy(perr[0])
                    + r")\cdot e^{(" + stringy(popt[1]) 
                    + r'\pm' + stringy(perr[1]) + r")\cdot x"
                    + r"+(" + stringy(popt[2])
                    + r'\pm' + stringy(perr[2]) + r")}$")
    elif f == "poisson":
        fit_label = (r"Curve Fit: $\lambda = (" + stringy(popt[0]) + r'\pm' 
                    + stringy(perr[0]) + r")$")
    # y values for plot
    y_pred = func(x, *popt)
    ax.plot(x, y_pred, label=fit_label)



def calc_cov(inputOT, limit):
    inpSTART  = [np.array(row[:-1]) for row in inputOT]
    inpEND    = [np.array(row[1:]) for row in inputOT]
    inpDIFF   = [inpEND[i] - inpSTART[i] for i in range(len(inputOT))]
    inp_mean  = np.array([np.mean(row) for row in inpDIFF])
    inp_std   = np.array([np.std(row) for row in inpDIFF])
    cov       = inp_mean / inp_std
    cov2 = []
    infs2 = []
    for i, val in enumerate(cov):
        if val<limit:
            cov2.append(val)
        else:           
            infs2.append(i)
    return np.array(cov2), np.array(infs2)



###############################################################################
############################## Plotting Functions #############################
###############################################################################


def indiExtended(indiNeuronsDetailed, threshM, recNum, flip=0):
    captiontxt = captiontxt_GLOBAL
    maxLen = len(indiNeuronsDetailed)
    showRange = recNum if maxLen > recNum else maxLen
    if flip:
        name_ext = ""
        indiNeuronsDetailed = np.flip(indiNeuronsDetailed)
    
    ### Set up Figures ###
    fig = plt.figure(constrained_layout=False,  figsize = (8,7))
    h_ratio = 10 - showRange / 2 if 10 - showRange / 2 > 2 else 2
    gs  = fig.add_gridspec(ncols=1, nrows=2,
                           height_ratios=[h_ratio, 1])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_yticks([])

    ### Check Size ###
    lengthOfLists = [len(row) for row in indiNeuronsDetailed]
    minMaxDiff = np.max(lengthOfLists) - np.min(lengthOfLists)
    if minMaxDiff:
        print("WARNING: Asymmetric Growth of individual neurons recorded")
        captiontxt += (f"\n  unequal size of neurons, difference between max" +
                       f" and min = {minMaxDiff}")
    lengthOfPlot = max(lengthOfLists)

    ### Set up Lines ###
    col = ['blue', 'green', 'red', 'purple', 'grey', 'black']
    labelNames = ["cum. positive", "net input", "cum. negative", 
                  "positive internal", "positive external", 'threshold']
    lstyle = ['-', '-', '-', ':', '--', '-']

    ### Draw Lines ###
    dataspike = []
    for i in range(showRange):
        rec = np.transpose(indiNeuronsDetailed[i])
        xs = range(0, len(rec[1]))
        # lines = [[] for x in range(showRange)]
        spike = [1 if rec[1][j] > rec[5][j] and rec[1][j - 1] < rec[5][j - 1]
                 else 0 for j in range(len(rec[1]))]
        # spike += [0.3 for x in range(lengthOfPlot - len(rec[1]))]
        dataspike.append(spike)
        for j in range(len(rec)):
            # lines[i].append(ax1.plot(xs, rec[j], color=col[j], linewidth=.8,
            #                      label=labelNames[j], linestyle=lstyle[j]))
            ax1.plot(xs, rec[j], color=col[j], linewidth=.8,
                     label=labelNames[j], linestyle=lstyle[j])
        rec_mean = np.mean(rec,axis=1)
        rec_std = np.std(rec,axis=1)
        print(rec_mean.shape)
        print(rec_std[2]/rec_mean[2])
        print(rec_std[3]/rec_mean[3])

    ax2.imshow(dataspike, aspect='auto', cmap='Greys', interpolation='nearest')
    


    ### Add Description ###
    labelX = "Time"
    plt.xlabel(labelX + '\n\n' + captiontxt_GLOBAL)
    # fig.suptitle('Individual Neuron Firing Pattern', fontsize=20)
    ax1.set(ylabel='Current')
    ax2.set(ylabel='Spike')
    # ax1.legend()#by_label.values(), by_label.keys())
    handles, labels = ax1.get_legend_handles_labels()
    # ax1.legend(
    #           handles, labels,
    #           bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #           ncol=2,
    #         #    mode="expand",
    #            borderaxespad=0.
    #           )
    # ax1.set_xticks([])
    xpos = -.01
    leg1 = Legend(ax1,
                [handles[0],handles[2]],
                [labels[0],labels[2]],
                bbox_to_anchor=(.6, xpos),
                # loc='lower right', 
                frameon=False)
    leg2 = Legend(ax1,
                [handles[1],handles[5]],
                [labels[1],labels[5]],
                bbox_to_anchor=(.2, xpos),
                borderaxespad=0.,
                # loc='lower right', 
                frameon=False)
    leg3 = Legend(ax1,
                [handles[3],handles[4]],
                [labels[3],labels[4]],
                bbox_to_anchor=(1.02, xpos),
                # loc='lower right', 
                frameon=False)

    plt.subplots_adjust(hspace=.3)
    ax1.add_artist(leg1)
    ax1.add_artist(leg2)
    ax1.add_artist(leg3)

    ### Save and Show ###
    folder = utils.checkFolder(figfolder_GLOBAL)
    name = "IndiExt"
    # fullname = utils.testTheName(folder +name+titletxt_GLOBAL , "png")
    fullname = utils.testTheName(folder + name, "pdf")
    if savefig_GLOBAL:
        plt.savefig(fullname)
        utils.plotMessage(fullname)
    if showPlots_GLOBAL:
        plt.show()
    plt.close(fig)


def newDistri(inputOT, timer):

    norm, bin_range, normal_weights = calc_firing(inputOT)

    fig = plt.figure(tight_layout=True)
    plt.hist(norm, bins=bin_range, weights=normal_weights)

    title_of_plot  = 'Firing Rate Distribution'
    xlabel_of_plot = ('Normalized Firing Rate' + '\n\n'
                      + captiontxt_GLOBAL 
                    #   + "\n" + disclaimer
                      )
    ylabel_of_plot  = 'Density'
    name_of_plot    = "Distri"
    finishplot(title_of_plot, xlabel_of_plot, ylabel_of_plot, name_of_plot, fig)

def calc_firing(inputOT):
    actiRowLen = np.array([len(row) for row in inputOT])
    norm       = actiRowLen / np.mean(actiRowLen)

    d = np.diff(np.unique(norm)).min()
    first_bin = norm.min() - float(d) / 2
    last_bin = norm.max() + float(d) / 2
    bin_range = np.arange(first_bin, last_bin + d, d)

    normal_weights = np.ones(len(norm)) / len(norm)
    return norm, bin_range, normal_weights

def fireCOV(inputOT, timer):
    norm, _ ,_ = calc_firing(inputOT)
    limit = 5
    cov,infs = calc_cov(inputOT, limit)
    norm = np.delete(norm, infs)
    fig = plt.figure(tight_layout=True)
    plt.scatter(cov,norm, s=.1)

    title_of_plot  = 'Firing Rate Distribution'
    xlabel_of_plot = ('Coefficient of Variation' + '\n\n'
                      + captiontxt_GLOBAL 
                      )
    ylabel_of_plot  = 'Normalized Firing Rate'
    name_of_plot    = "fireCOV"
    finishplot(title_of_plot, xlabel_of_plot, ylabel_of_plot, name_of_plot, fig)


def newMeanOT(mean_inputOT, isFire=0):
    def func(x, m, b):
        return m*x+b
    colors = ['b', 'r']
    labels = ['Excitatory', 'Inhibitory']
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    line = np.arange(np.array(mean_inputOT).shape[1])

    for i in range(2):
        ax.plot(mean_inputOT[i], label=labels[i], color=colors[i])
        regr("linear",line,mean_inputOT[i],ax)

    plt.legend()
    if isFire   :  title_of_plot   = 'Mean Rate of Fire over Time' 
    else        :  title_of_plot   = 'Mean Rate of Activation over Time' 
    if isFire   :  ylabel_of_plot  = 'Activation Rate'
    else        :  ylabel_of_plot  = 'Firing Rate'
    xlabel_of_plot  = 'Time' + '\n\n' + captiontxt_GLOBAL
    name_of_plot    = "meanOT"
    finishplot(title_of_plot, xlabel_of_plot, ylabel_of_plot, name_of_plot, fig)
  

def fit_func(t, a, b):
    return a * np.exp(b * t)


def newInterspike(inputOT, timer, display_Log=1):
    # matplotlib.rcParams['text.usetex'] = True
    inpSTART  = [np.array(row[:-1]) for row in inputOT]
    inpEND    = [np.array(row[1:]) for row in inputOT]
    inpDIFF   = [inpEND[i] - inpSTART[i] for i in range(len(inputOT))]
    flat_diff = np.array([x for row in inpDIFF for x in row])
    print(flat_diff.shape )
    flat_diff = [x for x in flat_diff if x <90]
    print(len(flat_diff))

    co_V = np.std(flat_diff)/ np.mean(flat_diff)
    print("coefficient")
    print(co_V)

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    bin_range = np.arange(min(flat_diff), max(flat_diff), 0.5)
    # bin_range = np.arange(timer)
    weights_ = np.ones(len(flat_diff)) / len(flat_diff)
    n, binz, _ = ax.hist(flat_diff, bins=bin_range, weights=weights_)
    name_log = ''
    if display_Log:
        name_log = '_logscale'
        plt.yscale('log', nonposy='clip')
    bincenters = 0.5 * (binz[1:] + binz[:-1])
    # print(bincenters)
    peak_pos = np.argmax(n)
    regr("exp", bincenters, n, ax)
    cov_txt = 'Coefficient of Variation: %.3f' % co_V
    plt.plot([], [], ' ', label=cov_txt)

    ax.legend()
    name_of_plot    = "interspike" +name_log
    title_of_plot   = 'Interspike Interval' 
    xlabel_of_plot  = 'Time' + '\n\n' + captiontxt_GLOBAL
    ylabel_of_plot  = 'Density'
    finishplot(title_of_plot, xlabel_of_plot, ylabel_of_plot, name_of_plot, fig)


def Oneterspike(inputOT, timer, display_Log=1):
    # matplotlib.rcParams['text.usetex'] = True
    inpSTART  = [np.array(row[:-1]) for row in inputOT]
    inpEND    = [np.array(row[1:]) for row in inputOT]
    inpDIFF   = [inpEND[i] - inpSTART[i] for i in range(len(inputOT))]
    idx = 0
    One_diff  = inpDIFF[idx]
    cov, _ = calc_cov(inputOT)
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    bin_range = np.arange(min(One_diff), max(One_diff), 0.5)
    # bin_range = np.arange(timer)
    weights_ = np.ones(len(One_diff)) / len(One_diff)
    n, binz, _ = ax.hist(One_diff, bins=bin_range, weights=weights_)
    name_log = ''
    bincenters = 0.5 * (binz[1:] + binz[:-1])
    peak_pos = np.argmax(n)
    regr("exp", bincenters, n, ax)
    cov_txt = 'Coefficient of Variation: %.3f' % cov[idx]
    plt.plot([], [], ' ', label=cov_txt)

    ax.legend()
    name_of_plot    = "Oneterspike" + name_log
    title_of_plot   = 'Interspike Interval' 
    xlabel_of_plot  = 'Time' + '\n\n' + captiontxt_GLOBAL
    ylabel_of_plot  = 'Density'
    finishplot(title_of_plot, xlabel_of_plot, ylabel_of_plot, name_of_plot, fig)


def covDIST(inputOT, timer, display_Log=0):
    # matplotlib.rcParams['text.usetex'] = True
    limit = 5
    cov, _ = calc_cov(inputOT, limit)
    cov = [x for x in cov if x<5]
    print(min(cov))
    print(max(cov))
    bin_range = np.arange(min(cov), max(cov),0.01)
    # bin_range = np.arange(timer)
    weights_ = np.ones(len(cov)) / len(cov)
    
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    n, binz, _ = ax.hist(cov, bins=bin_range, weights=weights_)
    name_log = ''
    if display_Log:
        name_log = '_logscale'
        plt.yscale('log', nonposy='clip')
    bincenters = 0.5 * (binz[1:] + binz[:-1])
    regr("exp", bincenters, n, ax)

    ax.legend()
    name_of_plot    = "interspike" +name_log
    title_of_plot   = 'Interspike Interval' 
    xlabel_of_plot  = 'Time' + '\n\n' + captiontxt_GLOBAL
    ylabel_of_plot  = 'Density'
    finishplot(title_of_plot, xlabel_of_plot, ylabel_of_plot, name_of_plot, fig)


def dots(inputOT, timer):
    precision = 1  # .1
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    val_OT = np.zeros((len(inputOT), int(timer / precision)))
    for i in range(len(inputOT)):
        for zeit in inputOT[i]:
            val_OT[i, int(zeit / precision)] = 1
    ax.imshow(val_OT, aspect='auto', cmap='Greys',
              extent=[0, timer, 0, len(inputOT)], interpolation='nearest')

    ### Kein Rand ###
    ax.set_ylim(ymin=0, ymax=len(inputOT))
    ax.set_xlim(xmin=0, xmax=timer)
    ### Keine Ticks
    ax.set_yticks([])
    ### Background Colour und Beschriftung ###
    plt.text(0.06, 0.7, "Inhibitory", fontsize=8, rotation=90,
             transform=plt.gcf().transFigure)
    plt.text(0.06, 0.36, "Excitatory", fontsize=8, rotation=90,
             transform=plt.gcf().transFigure)
    plt.axhspan(0, len(inputOT) / 2, facecolor='blue', alpha=0.3)
    plt.axhspan(len(inputOT) / 2,len(inputOT), facecolor='red', alpha=0.3)

    title_of_plot   = 'Neuronal Activity over Time' 
    xlabel_of_plot  = 'Time' + '\n\n' + captiontxt_GLOBAL
    ylabel_of_plot  = 'Neurons\n'
    name_of_plot    = "dots"
    finishplot(title_of_plot, xlabel_of_plot, ylabel_of_plot, name_of_plot, fig)


def mean_vs_ext(meanList, isFire=0):
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    pltColor = ['b','r']
    pltLabel = ['Excitatory','Inhibitory']
    lstyle   = ['-', '--']
    def func(x, m, b, ):
        return m * x + b
    for i in range(2):
        # ax.plot(meanList[0], meanList[i + 1], color=pltColor[i],
        #          label=pltLabel[i], linestyle=lstyle[i], marker='o')
        ax.plot(meanList[0], meanList[i + 1], color=pltColor[i],
                 label=pltLabel[i],linestyle='', marker='o')
        x, y = meanList[0], meanList[i + 1]
        popt, pcov = scipy.optimize.curve_fit(func, x, y)  # , p0=defaults)
        perr = np.sqrt(np.diag(pcov))

        fit_label = (r"Curve Fit: $m \cdot  x+b"
                    + f"$\nm = $" + stringy2(popt[0], perr[0])
                    + f"$, b = $" + stringy2(popt[1], perr[1]) + "$")
        y_pred = func(x, *popt)
        ax.plot(x, y_pred, label=fit_label, linestyle=lstyle[i])
        # regr("linear", meanList[0], meanList[i + 1], ax)
    plt.legend()
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    title_of_plot   = '' 
    if isFire   :   ylabel_of_plot  = 'Mean of Firing Rate'
    else        :   ylabel_of_plot  = 'Mean of Activation Rate'
    xlabel_of_plot  = 'External Rate'
    name_of_plot    = "mean_vs_ext"
    finishplot(title_of_plot, xlabel_of_plot, ylabel_of_plot, name_of_plot, fig)


def describe(toDo, info, figs, add_describ):
    sizeMax = np.sum(info["sizeM"])
    np.set_printoptions(edgeitems=10)
    captiontxt = (f'Network Size: {sizeMax}  K: {info["K"]} \
                    mean_Ext: \ {info["meanExt"]} \n time: {info["timer"]}   \
                    jE: {info["jE"]}   jI: {info["jI"]}\n \
                    j_EE: {np.round(info["j_EE"], 3)},\
                    ext_E: {np.round(info["extM"][0])}')
    shorttxt = f'j_EE_{str(info["j_EE"])[:3]}_ext_E_{str(info["extM"][0])[:3]}'
    # f'_S{int(np.log10(sizeMax))}'\
    # + f'_K{int(np.log10(info["K"]))}_m{str(info["meanExt"])[2:]}\
    # _t{str(info["timer"])[:-1]}' # \njE: {jE}   jI: {jI} ' 

    if toDo["doRand"]:
        captiontxt += f",\n stochastic Updates"
        shorttxt += "_rY"
    else:
        captiontxt += ",\n deterministic Updates"
        shorttxt += "_rN"

    if (toDo["doThresh"] == "constant"):
        captiontxt += ", Thresholds = constant"
        shorttxt += "_tC"
    elif (toDo["doThresh"] == "gauss"):
        captiontxt += ", Thresholds = gaussian"
        shorttxt += "_tG"
    elif (toDo["doThresh"] == "bound"):  
        captiontxt += ", Thresholds = bounded"
        shorttxt += "_tB"

    ### still updating caption and title ###
    figfolder = info['figfolder'] + shorttxt 
    valueFolder = Path(str(info['valueFolder']) + shorttxt)
    if figs:
        global figfolder_GLOBAL 
        global captiontxt_GLOBAL
        global titletxt_GLOBAL   
        figfolder_GLOBAL  = figfolder
        if  add_describ:
            captiontxt_GLOBAL = captiontxt
        titletxt_GLOBAL   = shorttxt
        return [figfolder, shorttxt, captiontxt]
    else:
        return valueFolder


def plot_center(activeOT, fireOT, indiNeuronsDetailed,
                 info, drw, toDo):
    """ 
    Legacy 
    """
    threshM, timer = info["threshM"], info["timer"],
    sizeM, display_count = info["sizeM"], info["display_count"]
    add_describ = 0
    describe(toDo, info, 1, add_describ)
    ### Analysis ###
    mean_actiOT = eva.analyzeMeanOT(activeOT, sizeM)

    ### Plotting Routine ###
    if drw["pIndiExt"]:
        indiExtended(indiNeuronsDetailed, threshM, display_count)
    if drw["pIndiExt"]:
        indiExtended(indiNeuronsDetailed, threshM, display_count, 1)
    if drw["nDistri"]:
        newDistri(activeOT, timer)
    if drw["newMeanOT"]:
        newMeanOT(mean_actiOT)
    if drw["dots"]:
        dots(activeOT, timer)
    if drw["nInter_log"]:
        newInterspike(fireOT, timer)
    if drw["nInter"]:
        newInterspike(fireOT, timer, 0)
    if drw["Onter"]:
        print("onter")
        Oneterspike(fireOT, timer)
    if drw["cov_dist"]:
        print("cov dist")
        covDIST(fireOT, timer)
    if drw["fire_COV"]:
        fireCOV(fireOT, timer)

def the_draw():
    # Bools for if should be peotted or not
    pIndiExt    = 0
    nDistri     = 0
    newMeanOT   = 0
    nInter      = 0
    Onter       = 0
    nInter_log  = 0
    dots        = 0
    cov_dist    = 0             
    fire_COV    = 1
    drw = locals()
    return drw

def Vanilla():
    # mode_list = ['count_up', 'static', 'permute', 'rand']
    mode = "count_up"
    name = 'classic' + "_" + mode
    folderloc = "../ValueVault/testreihe_200313_1718j_EE_1_ext_E_1.0_rY_tC/"
    # folderloc = ""
    if folderloc:
        folderloc = Path(folderloc)
    else:
        folderloc = utils.setupFolder(name)[2]
        folderloc = folderloc.parent
        folderloc = Path( folderloc / utils.mostRecent(folderloc,name))

        if not folderloc.exists():
            name = 'testreihe'
            folderloc = utils.setupFolder(name)[2]
            folderloc = folderloc.parent
            folderloc = Path( folderloc / utils.mostRecent(folderloc))
    indiNeuronsDetailed, activeOT, fireOT, info, toDo = utils.recoverResults(folderloc)
    drw = the_draw()
    info['display_count'] = 1
    plot_center(activeOT, fireOT, indiNeuronsDetailed,
                 info, drw, toDo) 

def change_m0():

    loc = "m_vs_m0.npy"
    meanInput = np.load(loc)
    mean_vs_ext(meanInput)

if __name__ == '__main__':
    savefig_GLOBAL    = 0
    showPlots_GLOBAL  = 1

    # Select Script to run 
    Vanilla()
    # change_m0()
    pass