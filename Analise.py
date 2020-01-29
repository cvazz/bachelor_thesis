import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

import utils


### print interesting ###
def show_interesting_rows(orig, res):
    interes = []
    thresh_interes = .89
    for row in res:
        interes.append([])
        for j in range(len(row)):
            if j > 2 and row[j]>thresh_interes:
                interes[-1].append(j)
            # else:
            #     interes[-1].append(0)

    for oro, iro in zip(orig,interes):
        if iro:
            desc = "j_EE = %3.2f, ext_E = %3.2f: " % tuple(oro[:2])
            print(desc, end="")
            # print(oro[0])
            # print(oro[1])
            for it in iro:
                print(oro[it])

### PLOT ###
def drw_overview(res, jEE_first=0):
    plt.figure(tight_layout=True)
    cleg = []
    lleg = []

    ### style of plotted vars ###
    # print(col)
    color_length = 0
    for i in range(len(res)):
        if res[i][0] != res[i+1][0]:
            color_length = int(len(res)/(i+1))
            # print(color_length)
            break
    col = ['r', 'g', 'b', 'y']
    cmap = mpl.cm.plasma
    col = cmap(np.linspace(0, 0.9, color_length))
    lstyle = ['-', '-.', '--']

    ### plot all lines ###
    for j in range(3):
        lleg.append("delay: %d" %(j))
        for i in range(0, len(res), color_length):
            plo = np.transpose(res[i:i+color_length])
            plt.plot(plo[1], plo[j + 2], c=col[int(i / 4)], linestyle=lstyle[j])
            if j == 0:
                # print(plo[0])
                if (jEE_first):
                    cleg.append("J_EE: %3.2f" %(plo[0, 0]))
                else:
                    cleg.append("ext_E: %3.2f" % (plo[0, 0]))

    ### color predictive capabilities ###
    plt.axhspan(0.9, 1, facecolor='red', alpha=0.3)

    ### Custom Legend for less lines on display ###
    leg_elem = []
    for i in range(len(cleg)):
        line = mpl.lines.Line2D([0],[0],c=col[i], label=cleg[i],) 
        leg_elem.append(line)
    for i in range(len(lleg)):
        line = mpl.lines.Line2D([0],[0], c='grey', linestyle=lstyle[i], label=lleg[i],) 
        leg_elem.append(line)
    plt.legend(bbox_to_anchor=(1, 1.),  fontsize='small',handles = leg_elem) #

    ### Labels ###
    plt.ylabel('true prediction rate')
    if (jEE_first):
        plt.xlabel('ext_E')
    else:
        plt.xlabel("jEE")

    ### plot ###
    plt.show()

def drw_overviewOT(resOT, jEE_first=0):
    plt.figure(tight_layout=True)
    print(resOT.shape)
    res = np.mean(resOT,0)
    idx = np.arange(len(res[0,:]))
    idx[0], idx[1] = 1,0
    res = res[:,idx]
    print(res.shape)
    y_errors=np.std(resOT, 0, dtype=np.float64)
    y_errors = y_errors[:,idx]
    cleg = []
    lleg = []
    dist = len(res[0]) - 2
    ### style of plotted vars ###
    # print(col)
    color_length = 0
    for i in range(len(res)):
        if res[i][0] != res[i+1][0]:
            length_line = i+1
            amount_lines = int(len(res)/length_line)
            break

    print(length_line)
    print(amount_lines)
    col = ['r', 'g', 'b', 'y']
    cmap = mpl.cm.plasma
    col = cmap(np.linspace(0,.9,amount_lines))
    lstyle = ['-', '-.', '--']

    amount_of_lines = range(0, len(res), length_line)
    # start, end = 4,8
    # amount_of_lines = range(color_length*start, color_length*end, color_length)

    # print(y_errors)
    ### plot all lines ###
    for j in range(dist):
        lleg.append("delay: %d" %(j))
        for i in amount_of_lines:
            # pos = i*
            plo = np.transpose(res[i:i+length_line])
            pl_err = np.transpose(y_errors[i:i+length_line])
            plt.errorbar(plo[1],plo[j+2],yerr=pl_err[j+2],
                         capsize=8, c=col[int(i/length_line)], linestyle=lstyle[j])
            if j == 0: 
                # print(plo[0])
                if (jEE_first):
                    cleg.append("J_EE: %3.2f" %(plo[0,0]))
                else:
                    cleg.append("ext_E: %3.2f" %(plo[0,0]))

    ### color predictive capabilities ###
    plt.axhspan(0.9, 1, facecolor='red', alpha=0.3)

    ### Custom Legend for less lines on display ###
    leg_elem = []
    for i in range(len(cleg)):
        line = mpl.lines.Line2D([0],[0],c=col[i], 
        label=cleg[i],) 
        leg_elem.append(line)
    for i in range(len(lleg)):
        line = mpl.lines.Line2D([0],[0], c='grey', linestyle=lstyle[i], label=lleg[i],) 
        leg_elem.append(line)
    plt.legend(bbox_to_anchor=(1, 1.),  fontsize='small',handles = leg_elem ) #

    ### Labels ###
    plt.ylabel('true prediction rate')
    if (jEE_first):
        plt.xlabel('ext_E')
    else:
        plt.xlabel("jEE")

    ### plot ###
    plt.show()
def prepareTestData(res):
    x = []
    for i in range(5):
        xx = res[:,2:] + 0.1*(i-2)
        x.append( np. concatenate((res[:,:2], xx),1))
    return np.asarray(x)

def transform(res):
    np.set_printoptions(precision=3,suppress= True)
    a = (np.asarray(res.copy()))
    a = a[a[:,1].argsort()]
    idx = np.arange(len(a[0,:]))
    idx[0], idx[1] = 1,0
    a = a[:,idx]
    return a


def loadIt(name):
    orig = np.load(name, allow_pickle=True)
    numbers = orig[:,2:]
    total = [sum(arr) for row in numbers for arr in row]

    ### Convert Numbers into one relative value
    lis = []
    for row in numbers:
        lis.append([])
        for diff in row:
            lis[-1].append(diff[0]+diff[3])
    arr = np.array(lis)
    ### Join with label ###
    res = np.concatenate((orig[:,:2],arr/total[0]),1)
    return res


def loadItOT(name):
    orig = np.load(name, allow_pickle=True)
    numbers = orig[:,2:]
    total = [sum(arr) for row in numbers for arr in row]

    ### Convert Numbers into one relative value
    lis = []
    for row in numbers:
        lis.append([])
        for diff in row:
            lis[-1].append(diff[0]+diff[3])
    arr = np.array(lis)
    ### Join with label ###
    res = np.concatenate((orig[:,:2],arr/total[0]),1)
    return res

def main():
    ### Extract Numbers ###
    np.set_printoptions(precision=3,suppress= True)
    res = loadIt("test_the_machine.npy")
    res3 = np.load("test_the_machine_0129.npy")
    # print(res)
    # print(res.shape)

    # show_interesting_rows(orig,res)
    # drw_overview(res)
    # res2 = transform(res)
    # res3 = prepareTestData(res)
    # print(res3)
    drw_overviewOT(res3)

if __name__ == '__main__':
    main()