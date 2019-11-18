
"""
This is the main python file for my project
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

import neuron
import utils
import plots
import old

def afterSimulationAnalysis():
    useMostRecent = True
    vfolder = Path("ValueVault")
    if useMostRecent:
        loadTimeStr = utils.mostRecent(vfolder)
    else:
        loadTimeStr = "testreihe_191111_1038" #create a "find most recent" function
    valueFolder =  vfolder / loadTimeStr

    timestr = time.strftime("%y%m%d_%H%M")
    figfolder = "figs/testreihe_" + timestr

    showRange = 15
    indiNeuronsDetailed, fireCount, nval_over_time, infoDict = recoverResults(valueFolder)
    total_times_one = utils.rowSums((nval_over_time))
    print(np.mean(total_times_one))
    print(np.shape(indiNeuronsDetailed))


    shorttxt    = infoDict["shorttxt"]
    captiontxt  = infoDict["captiontxt"]
    threshM     = infoDict["threshM"]
    timer       = infoDict["timer"]
    sizeM       = infoDict["sizeM"]

    ### Plotting Routine ###
    pActiDist   = 0
    pIndiExt    = 0
    pInterspike = 0
    pDots       = 0

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

if __name__ == "__main__":
    afterSimulationAnalysis()