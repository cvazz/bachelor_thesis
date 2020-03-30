import warnings
import numpy as np
from pathlib import Path
import pickle
import time


def mostRecent(vfolder,name2ignore='testreihe'):
    """
    Finds last added "testreihe" in given folder
    
    :param vfolder: folder to look for testreihe
    :type vfolder: class: 'pathlib.PosixPath'
    :return: name of "max" testreihe
    :rtype: string
    """
    paths = list(vfolder.glob('**'))
    names =  [path.name for path in paths]
    filtered = [name for name in names if name2ignore in name]
    return max(filtered)


def testTheName(name,fileEnding):
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
    print(pathname)
    if pathname.exists():
        return pathname
    else: 
        raise NameError("no file with this name exists")


def checkFolder(folder):
    if folder == "":
        warnings.warn("\n\nEmpty Folder Address\n")
    folderpath = Path(folder)
    if not folderpath.exists():
        folderpath.mkdir()
    return folder + "/"


def relMax(fireCount,showRange):
    return np.argpartition(fireCount, -1*showRange)[-1*showRange:]


def rowSums(matrix):
    total = []
    for row in matrix:
        total.append(sum(row))
    return total


def plotMessage(fullname):
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


def setupFolder(name="testreihe"):
    timestr = time.strftime("%y%m%d_%H%M")
    figfolder = "../figs/" + name + "_" + timestr
    valuefoldername = "../ValueVault/" + name + "_" + timestr
    valueFolder = Path(valuefoldername)
    return timestr, figfolder, valueFolder


###############################################################################
####################### Save and Recover Functions ############################
###############################################################################


def saveResults(valueFolder, indiNeuronsDetailed, 
                activeOT, fireOT, info, toDo):
    """
    Save recordings of output to file.
    :param valueFolder: Path to storage folder
    :type valueFolder: class: 'pathlib.PosixPath'
    :param indiNeuronsDetailed:  

    """
    if not valueFolder.exists():
        valueFolder.mkdir(parents=True)

    remember_list = [info, toDo]
    indiNametxt = "indiNeurons"
    infoNametxt = "infoDict"
    fireNametxt = "fireOT"      
    activeNametxt = "activeOT"    

    indiName = makeNewPath(valueFolder, indiNametxt, "npy")
    fireName = makeNewPath(valueFolder, fireNametxt, "npy")
    infoName = makeNewPath(valueFolder, infoNametxt, "pkl")
    activeName = makeNewPath(valueFolder, activeNametxt, "npy")

    np.save(indiName, indiNeuronsDetailed)
    np.save(fireName, fireOT)
    np.save(activeName, activeOT)
    infoName.touch()
    with open(infoName, "wb") as infoFile:
        pickle.dump(remember_list, infoFile, protocol=pickle.HIGHEST_PROTOCOL)


def recoverResults(valueFolder):
    """
    Load saved results from file.

    Missing functionality: so far, does only take neurons with preset names

    :param valueFolder: |valueFolder_desc|
    :return: indiNeuronsDetailed,  
    """

    indiNametxt     = "indiNeurons"
    infoNametxt     = "infoDict"
    activeNametxt   = "activeOT"    
    fireNametxt     = "fireOT"      

    indiName        = makeExistingPath(valueFolder, indiNametxt, "npy")
    activeName      = makeExistingPath(valueFolder, activeNametxt, "npy")
    fireName        = makeExistingPath(valueFolder, fireNametxt, "npy")
    infoName        = makeExistingPath(valueFolder, infoNametxt, "pkl")

    indiNeuronsDetailed = np.load(indiName, allow_pickle=True)
    activeOT            = np.load(activeName, allow_pickle=True)
    fireOT              = np.load(fireName, allow_pickle=True)
   
    with open(infoName, 'rb') as infoFile:
        info, toDo = pickle.load(infoFile)

    return indiNeuronsDetailed, activeOT, fireOT, info, toDo

