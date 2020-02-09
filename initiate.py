###############################################################################
############################## Imported Modules ###############################
###############################################################################

### Numbers ###
import numpy as np
import matplotlib.pyplot as plt

### File Interaction and Manipulation ###
from pathlib import Path

### Duh ###
import time
import warnings
### Machine Learning
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import PassiveAggressiveClassifier

### Local Modules ###
import neuron as nn
import mathsim as msim
import utils
import plots

print_GLOBAL = 1

###############################################################################
########################### Utility Functions ###################################
###############################################################################
# 

def useless(*args):
    pass
    

def check_plt():
    fig = plt.figure()
    plt.close(fig)


def stamp(timestamp, name):
    timestamp.append([name, time.time(), time.time() - timestamp[-1][1]])
    print (timestamp[-1][0], end= ": ")
    utils.timeOut(timestamp[-1][2])


def analyzeMeanOT(inputOT, sizeM):
    # Get an array of ints for excitatory and inhibitory,
    # each int represents one activation/firing at this time point
    # int is only one level of precision. It could be modified
    flat = [np.array([x for row in inputOT[:sizeM[0]] for x in row], dtype=int),
            np.array([x for row in inputOT[sizeM[0]:] for x in row], dtype=int)]
    # Count how many individuals have fired at a given time
    uniq = [np.unique(act, return_counts=1) for act in flat]

    # Convert count above into a value between 0 and 1
    normalized = [uniq[i][1] / sizeM[i] for i in range(2)]

    return normalized


def transform2binary(xOT, timer):
    """Transforms information of spike times into binary record of spikes

    Top-level is time or in other words a list of spikes at time i. The index of a one
    marks the location of the spike
    :param xOT: [description]
    :type xOT: [type]
    :param timer: [description]
    :type timer: [type]
    :return: [description]
    :rtype: [type]
    """
    intOT = [np.trunc(row) for row in xOT]
    output = []
    for row in intOT:
        output.append([])
        for i in range(timer):
            output[-1].append(int(i in row))
    return np.transpose(output)


def abstand(x, y, max_):
    xx = []
    yy = []
    max_delay = max_ - 1
    for i in range(max_delay):
        xx.append(x[i : -max_delay + i])
        yy.append(y[: -max_delay])
    xx.append(x[max_delay:])
    yy.append(y[:-max_delay])
    return xx, yy


def comp(label, estimate):
    accu = np.zeros((4), np.int64)

    t = np.zeros((2, 2))
    for i in range(len(label)):
        pos = label[i] + estimate[i] * 2
        accu[0] += label[i] and estimate[i]
        accu[1] += (not label[i]) and estimate[i]
        accu[2] += label[i] and not estimate[i]
        accu[3] += not label[i] and not estimate[i]
        t[label[i]][estimate[i]] += 1
    total = np.sum(accu)
    accu_str = ( "true positive", "false_alarm", "miss \t", "true negative",)
    if print_GLOBAL:
        for i in range(len(accu)):
            print(accu_str[i] + ":\t" + str(accu[i]/total))
    return accu


def split_train_test(x, y, split_point):
    """ Splits two arrays of size (N x rows x beliebig) in roughly half

    :param x: [description]
    :type x: [type]
    :param y: [description]
    :type y: [type]
    :param split_point: [description]
    :type split_point: [type]
    :return: [description]
    :rtype: [type]
    """
    split = int(len(x[0]) * split_point)
    x_a, y_a, x_b, y_b = [], [], [], []
    for x_spalte, y_spalte in zip(x,y):
        x_a.append(x_spalte[:split])
        x_b.append(x_spalte[split:])
        y_a.append(y_spalte[:split])
        y_b.append(y_spalte[split:])

    return x_a, y_a, x_b, y_b


###############################################################################
########################### Setup Functions ###################################
###############################################################################


def run_box(jCon, thresh, external, info, toDo):
    """
    executes the differen sequences
    """
    # info['timestr']
    if print_GLOBAL:
        print("run")
    timestart = time.time()
    valueFolder = describe(toDo, info, 0)

    ### The function ###
    nval = nn.createNval(info["sizeM"], info["meanStartActi"]) 
    indiNeuronsDetailed = [[] for i in range(info['recNum'])]
    activeOT, fireOT, labels = nn.update_org(nval, jCon, thresh, external,
                                             indiNeuronsDetailed, toDo, info)

    ### time check ###
    timeend = time.time()
    if print_GLOBAL:
        print("runtime of routine")
        utils.timeOut(timeend - timestart)
    if not toDo["switch"]:
        nn.saveResults(valueFolder, indiNeuronsDetailed, 
                       activeOT, fireOT, info, toDo)
    # (indiNeuronsDetailed, activeOT, fireOT, nval_OT, info, toDo
    # )= nn.recoverResults(valueFolder)
    return (indiNeuronsDetailed, activeOT, fireOT, labels)


def describe(toDo, info, figs):
    sizeMax = np.sum(info["sizeM"])
    np.set_printoptions(edgeitems=10)
    captiontxt = (f'Network Size: {sizeMax}  K: {info["K"]}  mean_Ext: {info["meanExt"]} \n\
                    time: {info["timer"]}   jE: {info["jE"]}   jI: {info["jI"]}\n\
                    j_EE: {np.round(info["j_EE"],3)}, ext_E: {np.round(info["extM"][0])}')
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
    threshM, timer, sizeM, display_count = info["threshM"], info["timer"], info["sizeM"], info["display_count"]
    describe(toDo, info, 1)
    ### Analysis ###
    mean_actiOT = analyzeMeanOT(activeOT, sizeM)

    ### Plotting Routine ###
    if drw["pIndiExt"]:
        plots.indiExtended(indiNeuronsDetailed, threshM, display_count)
    if drw["pIndiExt"]:
        plots.indiExtended(indiNeuronsDetailed, threshM, display_count, 1)
    if drw["nDistri"]:
        plots.newDistri(activeOT, timer)
    if drw["newMeanOT"]:
        plots.newMeanOT(mean_actiOT)
    if drw["dots2"]:
        plots.dots2(activeOT, timer)
    if drw["nInter_log"]:
        plots.newInterspike(fireOT, timer)
    if drw["nInter"]:
        plots.newInterspike(fireOT, timer, 0)


def predic_readout(activeOT, labels_train, info):
    ### Learning Part ###
    timestart = time.time()
    dist  = info["dist"]
    input_train = transform2binary(activeOT, info["timer"])
    model_A = [LogisticRegression(solver="lbfgs", max_iter=160) for _ in range(dist)]
    x_A, y_A = abstand(input_train, labels_train, dist)
    xtrain_A, ytrain_A, xtest_A, ytest_A = split_train_test(x_A, y_A, 0.5)
    for i in range(dist):
        model_A[i].fit(xtrain_A[i], ytrain_A[i])

    timeend = time.time()
    if print_GLOBAL:
        utils.timeOut(timeend - timestart)

    ### Test & Evaluate ###
    timestart = time.time()

    estimate_A = [model_A[i].predict(xtest_A[i]) for i in range(dist)]
    readout_ = [comp(ytest_A[i],estimate_A[i]) for i in range(dist)]

    timeend = time.time()
    if print_GLOBAL:
        utils.timeOut(timeend - timestart)
    ### Convert Readout to single precision value ###
    correctness = [(delay[0]+delay[3])/sum(delay) for delay in readout_]
    return correctness


##############################################################################
################################## Main Area #################################
##############################################################################


def changeExt_():
    check_plt()
    info = numParam()
    toDo = doParam()[1]
    info["meanExt"] = 0.04
    jCon = nn.createjCon(info)
    thresh = nn.createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
    mE_List = [0.04, 0.1, 0.2, 0.3]
    meanList = []
    meanList2 = []
    for i in range(len(mE_List)):
        external = nn.createExt(info, mE_List[i])     
        activeOT = run_box( jCon, thresh, external,  info, toDo)[1] #1:active, 2:fire
        means = analyzeMeanOT(activeOT, info["sizeM"])
        meanList.append([mE_List[i]])
        meanList2.append([mE_List[i]])
        meanList[-1] += [np.mean(means[i][10:])for i in range(2)]
        meanList2[-1] += [np.mean(means[i])for i in range(2)]
    meanList = np.transpose(meanList)
    meanList2 = np.transpose(meanList2)
    print(meanList)
    print(meanList2)
    plots.figfolder_GLOBAL = info["figfolder"]
    plots.mean_vs_ext(meanList)


def changeThresh_():
    check_plt()
    ### Specify Parameters
    info = numParam()
    (drw, toDo) = doParam()

    ### Create constant inputs to function
    jCon = nn.createjCon(info)
    external = nn.createExt(info["sizeM"],info["extM"], info["K"], info["meanExt"])     

    doThresh = ["constant", "gauss", "bound"]
    for i in range(len(doThresh)):
        toDo["doThresh"] = doThresh[i]
        thresh = nn.createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
        (indiNeuronsDetailed,  
         activeOT, fireOT, _ #label
        ) = run_box(jCon, thresh, external, info, toDo,)

        plot_machine(
            activeOT, fireOT, indiNeuronsDetailed,
            info, drw, toDo)


def Machine(jEE=1, extE=1, dist=3):
    check_plt()
    ### Get Standard Parameters ###
    info = numParam()
    drw, toDo = doParam()

    ### Specify Parameters ###
    toDo["switch"] = 1
    info["j_EE"] = jEE
    info["extM"][0] = extE
    info["dist"] = dist
    info['recNum'] = 100#info['sizeM'][0]

    ### Create constant inputs to function ###
    jCon = nn.createjCon(info)
    external = nn.createExt(info)
    thresh = nn.createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])

    ### Run System ###
    (indiNeuronsDetailed, activeOT, fireOT, labels_train
    ) = run_box(jCon, thresh, external, info, toDo)
    activeOT = np.array(activeOT)
    ###
    correctness = predic_readout(activeOT, labels_train, info)
    overall_activity = activeOT.size/info['timer']/np.sum(info['sizeM'])
    ext_contrib = np.sum(indiNeuronsDetailed[:][:][4])
    int_contrib = np.sum(indiNeuronsDetailed[:][:][3])
    ratio_ext_int = ext_contrib / int_contrib
    useless(indiNeuronsDetailed, fireOT)
    return correctness, overall_activity, ratio_ext_int


def Vanilla():
    check_plt()
    timestamp = [["start", time.time(), 0]]

    ### Specify Parameters
    info = numParam()
    # info['timer'] = 10 #info['sizeM'][0]
    (drw, toDo) = doParam()
    toDo["switch"] = 0
    stamp(timestamp, "Params") 

    ### Create constant inputs to function
    jCon = nn.createjCon(info)
    stamp(timestamp, "jCon") 
    external = nn.createExt(info)     
    stamp(timestamp, "external") 
    thresh = nn.createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
    stamp(timestamp, "thresh") 

    ### Updating Function
    (indiNeuronsDetailed, activeOT, fireOT, label
    ) = nn.run_update(jCon, thresh, external, info, toDo,)
    stamp(timestamp, "update cycle") 

    print(np.array(indiNeuronsDetailed).shape)
    plot_machine(
        activeOT, fireOT, indiNeuronsDetailed,
        info, drw, toDo)
    stamp(timestamp, "all plots") 
    ext_contrib = np.sum(indiNeuronsDetailed[:][:][4])
    int_contrib = np.sum(indiNeuronsDetailed[:][:][3])
    print("Share of Internal vs external Contribution")
    print(int_contrib / ext_contrib)
    print(test_list(indiNeuronsDetailed))
    useless(label)


def cust_save(name, predic, activity, ratio, string, info=None, toDo=None):
    ### Setup File Names ###
    folder = utils.checkFolder("data/" + name + time.strftime("%m%d"))
    info_name = "info"
    info_path = folder + info_name
    info_path = utils.testTheName(info_path, "txt")

    predic_name = "prediction"
    pred_path = folder + predic_name
    pred_path = utils.testTheName(pred_path, "npy")

    act_name = "activity"
    act_path = folder + act_name
    act_path = utils.testTheName(act_path, "npy")

    rat_name = "ratio"
    rat_path = folder + rat_name
    rat_path = utils.testTheName(rat_path, "npy")

    ### Save Array ###
    np.save(pred_path, predic)
    np.save(act_path, activity)
    np.save(rat_path, ratio)

    ### Get System Info ###
    if not info:
        info = numParam()
    if not toDo:
        toDo = doParam()[1]
    paramX = ["doThresh"]
    for param in paramX:
        if param in toDo:
            info[param] = toDo[param]

    ### Save Comment and System Info
    f = open(info_path,"w")
    f.write(string)
    f.write("\n")
    for key, val in info.items():
        f.write(key+ ", ")
        f.write(str(val)+ "\n")
    f.close()


def test_the_machine():
    def warnMe(jEE, extE, warn_me):
        print("this is the warning at: ", end="")
        print(extE, end=", ")
        print(jEE)
        print(warn_me[-1].category)
        print(warn_me[-1].message)
    ### Turn Off Printing
    plots.print_GLOBAL = 0
    global print_GLOBAL
    print_GLOBAL = 0
    nn.print_GLOBAL = 0

    repetitions = 5
    dist = 8
    range_extE = np.linspace(0.75, 1.2, 6)
    range_jEE = np.linspace(.9, 1.4, 12)

    timestart = time.time()
    readout_OT = []
    overall_OT = []
    ratio_OT = []
    record_warnings = []
    with warnings.catch_warnings(record=True) as warn_me:
        last_warn = 0
        for _ in range(repetitions):
            predic_readout = []
            overall_readout = []
            ratio_readout = []
            for extE in range_extE:
                for jEE in range_jEE:
                    print(f"jEE: {np.round(jEE,2)}, extE: {np.round(extE,2)}")
                    predic, overall, ratio = Machine(jEE, extE, dist)
                    predic_readout.append([jEE, extE, *predic])
                    overall_readout.append([jEE, extE, overall])
                    ratio_readout.append([jEE, extE, ratio])
                    if warn_me and warn_me[-1] != last_warn:
                        record_warnings.append(warn_me[-1])
                        last_warn = warn_me[-1]
                        warnMe(jEE, extE, warn_me)
            readout_OT.append(predic_readout)
            overall_OT.append(overall_readout)
            ratio_OT.append(ratio_readout)

    print(record_warnings)
    string = ("Axis 0: Repeat under same Conditions\n"+
              "Axis 1: Differing Conditions\n"+
              "Axis 2: j_EE, ext_E, delay= 0-"+str(dist))
    cust_save("vary_jEE_extE_", readout_OT,overall_OT, ratio_OT, string)
    # cust_save("overall_active_jEE_extE_", overall_OT, string)
    # cust_save("ratio_jEE_extE_", ratio_OT, string)
    timeend = time.time()
    utils.timeOut(timeend - timestart)

###############################################################################
############################# Customize Here ##################################
###############################################################################


def setupFolder():
    timestr = time.strftime("%y%m%d_%H%M")
    figfolder = "../figs/testreihe_" + timestr
    valuefoldername = "../ValueVault/testreihe_"
    valueFolder = Path(valuefoldername + timestr)
    return timestr, figfolder, valueFolder


def numParam():
    """
    Sets all parameters relevant to the simulation

    For historic reasons also sets the folder where figures and data are saved
    """

    timestr, figfolder, valueFolder = setupFolder()
    j_EE            = 1
    j_EI            = 1
    extM            = np.array([1,0.8])
    jE              = 2.
    jI              = 1.8
    threshM         = np.array([1., 0.7])
    tau             = 0.9
    meanExt         = 0.1
    meanStartActi   = meanExt
    recNum          = 100
    display_count   = 1
    ### Most changed vars ###
    timer           = 220
    K               = 1000
    size            = 4000
    sizeM           = np.array([size,size])

    info = locals()
    info["meanExt_M"] = np.array([.1, .3])
    # info["GUI"] = 0
    info.pop("size")
    return info


def doParam():
    """
    specifies most behaviors of 
    """
    #Bools for if should be peotted or not
    pIndiExt    = 1
    nDistri     = 1
    newMeanOT   = 1
    nInter      = 1
    nInter_log  = 1 
    dots2       = 1
    drw = locals()
    
    doThresh    = "constant" #"constant", "gaussian", "bounded"
    switch      = 0     #change external input?
    doRand      = 0     #Only one Sequence per Routine

    toDo = {}
    for wrd in ("doThresh", "doRand","switch"):
        toDo[wrd] = locals()[wrd]

    plots.savefig_GLOBAL    = 1
    plots.showPlots_GLOBAL  = 0
    return drw, toDo

def test_list(x):
    if isinstance(x,list):
        return test_list(x[0]) + 1
    else:
        return 0

if __name__ == "__main__":
    # changeExt_()   
    # changeThresh_()
    Vanilla()
    # test_the_machine()
    # Machine()
    pass

###############################################################################
################################## To Dos #####################################
###############################################################################
"""
meanExt change what happens, linear read out durch logisitic regression oder pseudo inverse
Trainieren auf entscheidung jetzt, in 1, in 2...
immer 2 Neuronen pro Zeit
input 1 Zeitschritt 1 oder 0 übergeben
Plots: IndiNeuronPlot: untere Linie Peaks nicth korrekt
Plots: Distri Fit

Statistik 5 mal wiederholen
Sanity Check
Prozess optimieren
mehrere Epochs
Input Verhältnis Excitatory zu Extern
"""