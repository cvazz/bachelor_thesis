import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

import utils

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]



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
    lstyle = np.array(linestyle_tuple)[:,1]#['-', '-.', '--']

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
    def prepareData(resOT):
        res = np.mean(resOT,0)
        idx = np.arange(len(res[0,:]))
        idx[0], idx[1] = 1,0
        res = res[:,idx]
        y_errors = np.std(resOT, 0, dtype=np.float64)
        y_errors = y_errors[:,idx]
        dist = len(res[0]) - 2
        return res, y_errors, dist
    ### style of plotted vars ###
    # print(col)
    def prepareColor(res):
        for i in range(len(res)):
            if res[i][0] != res[i+1][0]:
                length_line = i+1
                amount_lines = int(len(res)/length_line)
                break

        cmap = mpl.cm.plasma
        col = cmap(np.linspace(0,.9,amount_lines))
        return col, length_line

    ### plot all lines ###
    def actualPlot_mean(ax, res, y_errors, length_line, lstyle, jEE_first):
        cleg = []
        lleg = []
        amount_of_lines = range(0, len(res), length_line)
        for j in range(dist):
            lleg.append("delay: %d" %(j))
            for i in amount_of_lines:
                plo = np.transpose(res[i:i+length_line])
                pl_err = np.transpose(y_errors[i:i+length_line])
                if ax:
                    ax.errorbar(plo[1],plo[j+2],yerr=pl_err[j+2],
                                capsize=8, c=col[int(i/length_line)], linestyle=lstyle[j])
                else:
                    plt.errorbar(plo[1],plo[j+2],yerr=pl_err[j+2],
                                capsize=8, c=col[int(i/length_line)], linestyle=lstyle[j])
                if j == 0:
                    # print(plo[0])
                    if (jEE_first):
                        cleg.append("J_EE: %3.2f" %(plo[0,0]))
                    else:
                        cleg.append("ext_E: %3.2f" %(plo[0,0]))
        return cleg, lleg
    def actualPlot_std(ax,res, y_errors, length_line, lstyle, jEE_first):
        cleg = []
        lleg = []
        amount_of_lines = range(0, len(res), length_line)
        for j in range(dist):
            lleg.append("delay: %d" %(j))
            for i in amount_of_lines:
                plo = np.transpose(res[i:i+length_line])
                pl_err = np.transpose(y_errors[i:i+length_line])
                ax.plot(plo[1],pl_err[j+2],
                         c=col[int(i/length_line)], linestyle=lstyle[j])
                if j == 0:
                    # print(plo[0])
                    if (jEE_first):
                        cleg.append("J_EE: %3.2f" %(plo[0,0]))
                    else:
                        cleg.append("ext_E: %3.2f" %(plo[0,0]))
        return cleg, lleg
    ### color predictive capabilities ###
    # plt.axhspan(0.9, 1, facecolor='red', alpha=0.3)

    ### Custom Legend for less lines on display ###
    def make_leg(fig,cleg, lleg, lstyle):
        leg_elem = []
        for i in range(len(cleg)):
            line = mpl.lines.Line2D([0],[0],c=col[i], label=cleg[i])
            leg_elem.append(line)
        for i in range(len(lleg)):
            line = mpl.lines.Line2D([0],[0], c='grey', linestyle=lstyle[i], label=lleg[i])
            leg_elem.append(line)
        fig.legend(bbox_to_anchor=(1., .5), fontsize='small', handles=leg_elem)


    def make_labels(jEE_first):
        plt.ylabel('true prediction rate')
        if (jEE_first):
            plt.xlabel('ext_E')
        else:
            plt.xlabel("jEE")
    def make_labels_subplots(ax1,jEE_first):
        ax1.set_ylabel('true prediction rate')
        ax2.set_ylabel('standard deviation')
        if (jEE_first):
            plt.xlabel('ext_E')
        else:
            plt.xlabel("jEE")


    ### Acutal Function Body ###
    res, y_errors, dist = prepareData(resOT)
    col, length_line = prepareColor(res) 
    lstyle = np.hstack((np.array(['-', '-.', '--', '.']),
                        np.array(linestyle_tuple)[:,1]))
    plt_deviation = 0
    if plt_deviation:
        fig3 = plt.figure(constrained_layout=True)
        ax1 = fig3.add_subplot(211)
        ax2 = fig3.add_subplot(212)
        cleg, lleg = actualPlot_mean(ax1, res, y_errors, length_line, lstyle, jEE_first)
        cleg, lleg = actualPlot_std(ax2, res, y_errors, length_line, lstyle, jEE_first)
        make_leg(fig3, cleg, lleg, lstyle)
        make_labels_subplots(ax1,jEE_first)
    else:
        fig = plt.figure(tight_layout=True, figsize=(8,6))
        cleg, lleg = actualPlot_mean(None, res, y_errors, length_line, lstyle, jEE_first)
        make_leg(fig, cleg, lleg, lstyle)
        make_labels(jEE_first)
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
    res3 = np.load("data/vary_jEE_extE_0207/array.npy")
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