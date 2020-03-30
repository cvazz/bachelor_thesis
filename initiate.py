###############################################################################
############################## Imported Modules ###############################
###############################################################################

### Numbers ###
import numpy as np
import random
import matplotlib.pyplot as plt

### File Interaction and Manipulation ###
from pathlib import Path
import pickle

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
# import mathsim as msim
import utils
import plots
import evaluate as eva

print_GLOBAL = 1

###############################################################################
########################### Utility Functions #################################
###############################################################################


def useless(*args):
    pass


def check_plt():

    fig = plt.figure()
    plt.close(fig)


def stamp(timestamp, name):
    timestamp.append([name, time.time(), time.time() - timestamp[-1][1]])
    print(timestamp[-1][0], end=": ")
    utils.timeOut(timestamp[-1][2])


def analyzeMeanOT(inputOT, sizeM):
    """ Convert information of timepoints into mean activation per timestep
    """
    print(np.array(inputOT).shape)
    # Get an array of ints for excitatory and inhibitory,
    # each value represents one activation/firing at this time point
    flat = [np.array([np.floor(x) for row in inputOT[:sizeM[0]] for x in row]),
            np.array([np.floor(x) for row in inputOT[sizeM[0]:] for x in row])]
    
    highest = int(np.floor(max([max(f) for f in flat])))+1
    # Make sure, every value is accessed 
    flat_extra =[] 
    for i in range(len(flat)):
        flat_extra.append(np.concatenate((flat[i],np.arange(highest))))
    # Count how many individuals have fired at a given time
    uniq = [np.unique(act, return_counts=1) for act in flat_extra]
    # Substract added range to only count actual events
    true_counts = [uniq[i][1] - np.ones(highest) for i in range(2)]

    # Convert count above into a value between 0 and 1
    normalized = [true_counts[i] / sizeM[i] for i in range(2)]
    print(normalized)

    return np.array(normalized)


def transform2binary(xOT, timer):
    """Transforms information of spike times into binary record of spikes

    Top-level is time or in other words a list of spikes at time i. The index 
    of a one
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
    """ 
    takes in data x and label y and creates max_ sets of data which 
    are shifted by 0 up to 1- max_. This means that there datapoint 4 will
    be saved with label 4, 5, up to max_ - 1.
    """
    xx = []
    yy = []
    max_delay = max_ - 1
    for i in range(max_delay):
        xx.append(x[i:-max_delay + i])
        yy.append(y[: -max_delay])
    xx.append(x[max_delay:])
    yy.append(y[:-max_delay])
    return xx, yy


def comp(label, estimate):
    """Gives overview of prediction accuracy
    
    :param label: true information
    :type label: 1d Bool array
    :param estimate: machine's inferred information
    :type estimate: 1d Bool array
    :return: Information whether prediction was correct.
    :rtype: array.shape() = (4,)
    """
    accu = np.zeros((4), np.int64)

    t = np.zeros((2, 2))
    for i in range(len(label)):
        # pos = label[i] + estimate[i] * 2
        accu[0] += label[i] and estimate[i]
        accu[1] += (not label[i]) and estimate[i]
        accu[2] += label[i] and not estimate[i]
        accu[3] += not label[i] and not estimate[i]
        t[label[i]][estimate[i]] += 1
    total = np.sum(accu)
    accu_str = ("true positive", "false_alarm", "miss \t", "true negative",)
    if print_GLOBAL:
        for i in range(len(accu)):
            print(accu_str[i] + ":\t" + str(accu[i] / total))
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
    for x_spalte, y_spalte in zip(x, y):
        x_a.append(x_spalte[:split])
        x_b.append(x_spalte[split:])
        y_a.append(y_spalte[:split])
        y_b.append(y_spalte[split:])

    return x_a, y_a, x_b, y_b

def ml_save(name, predic, activity, ratio, internal, string, info=None, toDo=None):
    """ Saves results of ML function
    
    :param name: [description]
    :type name: [type]
    :param predic: [description]
    :type predic: [type]
    :param activity: [description]
    :type activity: [type]
    :param ratio: [description]
    :type ratio: [type]
    :param string: [description]
    :type string: [type]
    :param info: [description], defaults to None
    :type info: [type], optional
    :param toDo: [description], defaults to None
    :type toDo: [type], optional
    """
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

    int_name = "internal"
    int_path = folder + int_name
    int_path = utils.testTheName(int_path, "npy")
    ### Save Array ###
    np.save(pred_path, predic)
    np.save(act_path, activity)
    np.save(rat_path, ratio)
    np.save(int_path, internal)

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
    f = open(info_path, "w")
    f.write(string)
    f.write("\n")
    for key, val in info.items():
        f.write(key + ", ")
        f.write(str(val) + "\n")
    f.close()

###############################################################################
########################### Setup Functions ###################################
###############################################################################


def predic_readout(activeOT, labels_train, info):
    """ 
    Logistic Regression to estimate "differentness" between patterns of different labels
    Returns percentage of correct guesses for each delay.
    """
    ### Learning Part ###
    timestart = time.time()
    dist  = info["dist"]
    input_train = transform2binary(activeOT, info["timer"])
    model_A = [LogisticRegression(solver="lbfgs", max_iter=160)
               for _ in range(dist)]
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
    readout_ = [comp(ytest_A[i], estimate_A[i]) for i in range(dist)]

    timeend = time.time()
    if print_GLOBAL:
        utils.timeOut(timeend - timestart)
    ### Convert Readout to single precision value ###
    correctness = [(delay[0] + delay[3]) / sum(delay) for delay in readout_]
    return correctness


##############################################################################
################################## Main Area #################################
##############################################################################


def changeExt_():
    """ 
    

    """
    check_plt()
    name = "m_vs_m0"
    info = numParam(name)
    toDo = doParam()[1]
    info["meanExt"] = 0.04
    info["timer"] = 50
    jCon = nn.createjCon(info)
    thresh = nn.createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
    mE_List = np.arange(0.04, 0.32,.02)
    # mE_List = np.arange(0.04, 0.36,.08)
    meanList = []
    meanList2 = []
    for i in range(len(mE_List)):
        external = nn.createExt(info, mE_List[i])     
        activeOT = nn.run_update(jCon, thresh, external, info, toDo)[1]  # 1:active, 2:fire
        means = eva.analyzeMeanOT(activeOT, info["sizeM"])
        meanList.append([mE_List[i]])
        meanList2.append([mE_List[i]])
        meanList[-1] += [np.mean(means[i][10:])for i in range(2)]
        meanList2[-1] += [np.mean(means[i])for i in range(2)]
    meanList = np.transpose(meanList)
    meanList2 = np.transpose(meanList2)
    print(meanList)
    name = "m_vs_m0_K%d.npy" % info["K"]
    np.save(name, meanList)
    plots.figfolder_GLOBAL = info["figfolder"]
    plots.mean_vs_ext(meanList)


def changeThresh_():
    check_plt()
    ### Specify Parameters
    name = "thresh"
    info = numParam(name)
    (drw, toDo) = doParam()

    ### Create constant inputs to function
    jCon = nn.createjCon(info)
    external = nn.createExt(info)     

    doThresh = ["constant", "gauss", "bound"]
    for i in range(len(doThresh)):
        toDo["doThresh"] = doThresh[i]
        thresh = nn.createThresh(info["sizeM"], info["threshM"],
                                 toDo["doThresh"])
        (indiNeuronsDetailed, activeOT, fireOT, _  # label
        ) = nn.run_update(jCon, thresh, external, info, toDo,)

        plot_machine(
            activeOT, fireOT, indiNeuronsDetailed,
            info, drw, toDo)




def Machine(jEE=1, extE=1, dist=3):
    ### Get Standard Parameters ###
    info = numParam()
    _, toDo = doParam()

    ### Specify Parameters ###
    toDo["switch"] = 1
    info["j_EE"] = jEE
    info["extM"][0] = extE
    info["dist"] = dist
    info['recNum'] = 100  # info['sizeM'][0]

    ### Create constant inputs to function ###
    jCon = nn.createjCon(info)
    external = nn.createExt(info)
    thresh = nn.createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])

    ### Run System ###
    (indiNeuronsDetailed, activeOT, fireOT, labels_train
    ) = nn.run_update(jCon, thresh, external, info, toDo)
    activeOT = np.array(activeOT)
    ###
    correctness = predic_readout(activeOT, labels_train, info)
    contrib = np.mean(indiNeuronsDetailed[:info["recNum"]][:], axis = (0,1)) 
    overall_activity = contrib[3]
    ratio_ext_int = contrib[3]/contrib[4]
    useless(indiNeuronsDetailed, fireOT)
    return correctness, overall_activity, ratio_ext_int, internal

def compact_prediction():
    nn.print_GLOBAL = 0

    repetitions = 5
    dist = 10
    readout = []
    timestart = time.time()
    for i in range(repetitions):
        print("Round: %d"%i+1)
        predic, overall, ratio = Machine(1,1, dist)
        readout.append(predic)
    timeend = time.time()
    utils.timeOut(timeend - timestart)
    np.save("prediction_1_1.npy",readout)

def test_the_machine():
    def warnMe(jEE, extE, warn_me):
        print("this is the warning at: ", end="")
        print(extE, end=", ")
        print(jEE)
        print(warn_me[-1].category)
        print(warn_me[-1].message)
    ### Turn Off Printing
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
    internal_OT = []
    record_warnings = []
    with warnings.catch_warnings(record=True) as warn_me:
        last_warn = 0
        for _ in range(repetitions):
            predic_readout = []
            overall_readout = []
            ratio_readout = []
            internal_readout = []
            for extE in range_extE:
                for jEE in range_jEE:
                    print(f"jEE: {np.round(jEE,2)}, extE: {np.round(extE,2)}")
                    predic, overall, ratio, internal = Machine(jEE, extE, dist)
                    predic_readout.append([jEE, extE, *predic])
                    overall_readout.append([jEE, extE, overall])
                    ratio_readout.append([jEE, extE, ratio])
                    internal_readout.append([jEE, extE, internal])
                    if warn_me and warn_me[-1] != last_warn:
                        record_warnings.append(warn_me[-1])
                        last_warn = warn_me[-1]
                        warnMe(jEE, extE, warn_me)
            readout_OT.append(predic_readout)
            overall_OT.append(overall_readout)
            ratio_OT.append(ratio_readout)
            internal_OT.append(internal_readout)

    print(record_warnings)
    string = ("Axis 0: Repeat under same Conditions\n" +
              "Axis 1: Differing Conditions\n" +
              "Axis 2: j_EE, ext_E, delay= 0-" + str(dist))
    ml_save("vary_jEE_extE_", readout_OT, overall_OT, ratio_OT, internal_OT, string)
    timeend = time.time()
    utils.timeOut(timeend - timestart)


def plain_vanilla(dur, start, swi=0):
    info = numParam()
    info['timer'] = dur #info['sizeM'][0]
    info['meanStartActi'] = start
    (drw, toDo) = doParam()
    toDo["switch"] = swi

    ### Create constant inputs to function
    jCon = nn.createjCon(info)
    external = nn.createExt(info)
    thresh = nn.createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])

    ### Updating Function
    (indiNeuronsDetailed, activeOT, fireOT, label
    ) = nn.run_update(jCon, thresh, external, info, toDo,)

    return activeOT


def maximal_diff():
    dur = 10
    runs = 5
    data = []
    sizeM = numParam()["sizeM"]
    for _ in range(runs):
        start = int(np.random.uniform(0,1)>0.5)
        data.append(analyzeMeanOT(plain_vanilla(dur,start),sizeM))
    # print(data)
    np.save("ml4_data.npy",data)


def hamming():
    flip = 1

    name = "hamming3"
    info = numParam()
    (_, toDo) = doParam()

    jCon = nn.createjCon(info)
    external = nn.createExt(info)
    thresh = nn.createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
    info["timer"] = 50
    hamm_comb = []
    rand_dist = []
    for i in range(10):
        orig = nn.createNval(info["sizeM"], info["meanStartActi"])
        changed = nn.createNval(info["sizeM"], info["meanStartActi"])

        dist = np.mean(abs(orig - changed))
        rand_dist.append(dist)
    for control in range(2):
        orig_list = []
        chang_list = []
        dist_list = []
            
        for i in range(10):
            rand_seed = i+100
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            info["flip"] = 0
            (indiNeuronsDetailed, activeOT, fireOT, label
            ) = nn.run_update(jCon, thresh, external, info, toDo,)
            orig = (transform2binary(activeOT, info["timer"]))

            if not control:
                np.random.seed(rand_seed)
                random.seed(rand_seed)
                info["flip"] = flip
            (indiNeuronsDetailed, activeOT, fireOT, label
            ) = nn.run_update(jCon, thresh, external, info, toDo,)
            changed = (transform2binary(activeOT, info["timer"]))

            dist = np.mean(abs(orig - changed),axis=1)
            if control:
                dist = np.concatenate(([rand_dist[i]],dist))
            else:
                dist = np.concatenate(([flip/np.sum(info["sizeM"])],dist))
            orig_mean = np.concatenate(([info['meanStartActi']], np.mean(orig,axis=1)))
            chang_mean =np.concatenate(([info['meanStartActi']], np.mean(changed,axis=1)))

            orig_list.append(orig_mean)
            chang_list.append(chang_mean)
            dist = i
            dist_list.append(dist)
        hamming = np.array(dist_list)
        hamm_comb.append(hamming)
        print(hamming.shape)
    print(hamm_comb)
    # np.save(name+'.npy',hamm_comb)


def Vanilla():
    """ 
    Iterates over whole process once according to specifications in numParam and doParam
    Records results.
    """
    timestamp = [["start", time.time(), 0]]

    ### Specify Parameters
    # mode_list = ['count_up', 'static', 'permute', 'rand']
    mode = "rand"
    name = "classic" + "_" + mode
    info = numParam(name)
    (drw, toDo) = doParam()
    toDo["switch"] = 0
    toDo["update_mode"] = mode

    ### Create constant inputs to function
    jCon = nn.createjCon(info)
    external = nn.createExt(info)
    thresh = nn.createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
    stamp(timestamp, "setup done")

    ### Updating Function
    (indiNeuronsDetailed, activeOT, fireOT, label
    ) = nn.run_update(jCon, thresh, external, info, toDo,)
    stamp(timestamp, "calculations done") 

    ### Save Results
    valueFolder = plots.describe(toDo, info, 0, 1)
    utils.saveResults(valueFolder, indiNeuronsDetailed, 
                activeOT, fireOT, info, toDo)
    stamp(timestamp, "saving done")



###############################################################################
############################# Customize Here ##################################
###############################################################################


def numParam(foldername='No_save_intended'):
    """
    Sets all parameters relevant to the simulation

    For historic reasons also sets the folder where figures and data are saved
    """

    timestr, figfolder, valueFolder = utils.setupFolder(foldername)
    j_EE            = 1
    j_EI            = 1
    extM            = np.array([1, 0.7])
    jE              = 2.
    jI              = 1.8
    threshM         = np.array([1., 0.7])
    tau             = 0.9
    meanExt         = 0.1
    meanStartActi   = meanExt
    recNum          = 100
    display_count   = 1
    ### Most changed vars ###
    timer           = 2000
    K               = 1000
    size            = 4000
    sizeM           = np.array([size, size])

    info = locals()
    info["meanExt_M"] = np.array([.1, .3])
    # info["GUI"] = 0
    info.pop("size")
    return info


def doParam():
    """
    specifies most behaviors of
    """
    # Bools for if should be peotted or not
    pIndiExt    = 1
    nDistri     = 1
    newMeanOT   = 1
    nInter      = 1
    nInter_log  = 1
    dots        = 1
    drw = locals()
 
    doThresh    = "constant"  # "constant", "gaussian", "bounded"
    switch      = 0     # change external input?
    doRand      = 0     # Only one Sequence per Routine

    update_mode =  "rand"
    toDo = {}
    for wrd in ("doThresh", "doRand", "switch", "update_mode"):
        toDo[wrd] = locals()[wrd]

    plots.savefig_GLOBAL    = 1
    plots.showPlots_GLOBAL  = 0
    return drw, toDo


def test_list(x):
    if isinstance(x, list):
        return test_list(x[0]) + 1
    else:
        return 0
if __name__ == "__main__":
    # changeExt_()
    # changeThresh_()
    Vanilla()
    # test()
    # compact_prediction()
    # maximal_diff()
    # first_diff()
    # test_the_machine()
    # hamming()
    # Machine()
    pass

###############################################################################
################################## To Dos #####################################
###############################################################################
"""
meanExt change what happens, linear read out durch logisitic regression
oder pseudo inverse
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

# Reservoir computing 

