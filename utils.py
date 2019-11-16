
import numpy as np
from pathlib import Path
import pickle


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
