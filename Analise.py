import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils
# from matplotlib.lines import Line2D

# import utils

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



def drw_overviewOT(resOT, y_name, save_name) :
    plt_deviation=0, 
    alt=0,
    jEE_first=0
    def prepareData(resOT):
        res = np.mean(resOT, 0)
        idx = np.arange(len(res[0, :]))
        idx[0], idx[1] = 1, 0
        res = res[:, idx]
        y_errors = np.std(resOT, 0, dtype=np.float64)
        y_errors = y_errors[:, idx]
        dist = len(res[0]) - 2
        return res, y_errors, dist
    ### style of plotted vars ###
    # print(col)

    def prepareColor(res):
        for i in range(len(res)):
            if res[i][0] != res[i + 1][0]:
                length_line = i + 1
                amount_lines = int(len(res) / length_line)
                break

        cmap = mpl.cm.plasma
        col = cmap(np.linspace(0, .9, amount_lines))
        return col, length_line

    ### plot all lines ###
    def actualPlot_mean(ax, res, y_errors, length_line, lstyle, jEE_first):
        cleg = []
        lleg = []
        amount_of_lines = range(0, len(res), length_line)
        print(list(amount_of_lines))
        for j in range(dist):
        # for j in range(3):
            lleg.append("delay: %d" % (j))
            for i in amount_of_lines:
            # for i in [0, 24, 60]:
                plo = np.transpose(res[i:i + length_line])
                pl_err = np.transpose(y_errors[i:i + length_line])
                if ax:
                    ax.errorbar(plo[1], plo[j + 2], yerr=pl_err[j + 2],
                                capsize=8, c=col[int(i / length_line)],
                                linestyle=lstyle[j])
                else:
                    plt.errorbar(plo[1], plo[j + 2], yerr=pl_err[j + 2],
                                 capsize=8, c=col[int(i / length_line)],
                                 linestyle=lstyle[j])
                if j == 0:
                    # print(plo[0])
                    if (jEE_first):
                        cleg.append("J_EE: %3.2f" % (plo[0, 0]))
                    else:
                        cleg.append(r"$E_E$"+": %3.2f" % (plo[0, 0]))
        return cleg, lleg

    def actualPlot_std(ax, res, y_errors, length_line, lstyle, jEE_first):
        cleg = []
        lleg = []
        amount_of_lines = range(0, len(res), length_line)
        for j in range(dist):
            lleg.append("delay: %d" % (j))
            for i in amount_of_lines:
                plo = np.transpose(res[i:i + length_line])
                pl_err = np.transpose(y_errors[i:i + length_line])
                if ax:
                    ax.plot(plo[1], pl_err[j + 2],
                        c=col[int(i / length_line)], linestyle=lstyle[j])
                else:
                    plt.plot(plo[1], plo[j + 2],
                        c=col[int(i / length_line)], linestyle=lstyle[j])
                if j == 0:
                    # print(plo[0])
                    if (jEE_first):
                        cleg.append("J_EE: %3.2f" % (plo[0, 0]))
                    else:
                        cleg.append(r"$E_E$"+": %3.2f" % (plo[0, 0]))
        return cleg, lleg
    def actualPlot_simp(ax, res, length_line, lstyle, jEE_first):
        cleg = []
        lleg = []
        amount_of_lines = range(0, len(res), length_line)
        for j in range(dist):
            lleg.append("delay: %d" % (j))
            for i in amount_of_lines:
                plo = np.transpose(res[i:i + length_line])
                plt.plot(plo[1], plo[j + 2],
                    c=col[int(i / length_line)], linestyle=lstyle[j])
                if j == 0:
                    # print(plo[0])
                    if (jEE_first):
                        cleg.append("J_EE: %3.2f" % (plo[0, 0]))
                    else:
                        cleg.append(r"$E_E$"+": %3.2f" % (plo[0, 0]))
        return cleg, lleg
    ### color predictive capabilities ###
    # plt.axhspan(0.9, 1, facecolor='red', alpha=0.3)

    ### Custom Legend for less lines on display ###
    def make_leg(fig, cleg, lleg, lstyle):
        leg_elem = []
        for i in range(len(cleg)):
            line = mpl.lines.Line2D([0], [0], c=col[i], label=cleg[i])
            leg_elem.append(line)
        fig.legend(
            bbox_to_anchor=(.98, .36), 
            # fontsize='small', 
            handles=leg_elem, 
            # loc="lower right"
            )

    def make_labels(jEE_first):
        plt.ylabel(y_name)
        if (jEE_first):
            plt.xlabel(r'$E_E$"+"')
        else:
            plt.xlabel(r"$J_{EE}$")


    ### Acutal Function Body ###
    res, y_errors, dist = prepareData(resOT)
    col, length_line = prepareColor(res)
    lstyle = np.hstack((np.array(['-', '-.', '--', ':']),
                        np.array(linestyle_tuple)[:, 1]))
    if alt:
        fig = plt.figure(tight_layout=True, figsize=(8, 6))
        cleg, lleg = actualPlot_std(None, res, y_errors, length_line, 
                                    lstyle, jEE_first)
        make_leg(fig, cleg, lleg, lstyle)
        make_labels(jEE_first)

    else:
        if plt_deviation:
            fig3 = plt.figure(constrained_layout=True)
            ax1 = fig3.add_subplot(211)
            ax2 = fig3.add_subplot(212)
            cleg, lleg = actualPlot_mean(ax1, res, y_errors,
                                        length_line, lstyle, jEE_first)
            cleg, lleg = actualPlot_std(ax2, res, y_errors, length_line,
                                        lstyle, jEE_first)
            make_leg(fig3, cleg, lleg, lstyle)
        else:
            fig = plt.figure(tight_layout=True, figsize=(8, 6))
            cleg, lleg = actualPlot_mean(None, res, y_errors, length_line, 
                                        lstyle, jEE_first)
            make_leg(fig, cleg, lleg, lstyle)
            make_labels(jEE_first)

    folder = "../figs"
    name = save_name
    folder = utils.checkFolder(folder)
    fullname = utils.testTheName(folder + name, "pdf")
    plt.savefig(fullname)
    utils.plotMessage(fullname)
    plt.show()
    plt.close(fig)

def drw_small_overview(resOT, plt_deviation=0, alt=0,jEE_first=0):
    def prepareData(resOT):
        res = np.mean(resOT, axis=0)
        res = np.mean(res, axis=0)[2:]
        y_errors = np.std(resOT, axis=(0,1), dtype=np.float64)[2:]
        return res, y_errors,
    def Other_data(resOT):
        res = np.mean(resOT, axis=0)
        y_errors = np.std(resOT, axis=(0), dtype=np.float64)
        print(y_errors)
        return res, y_errors

    ### Acutal Function Body ###
    if len(resOT.shape)>2:
        res, y_errors = prepareData(resOT)
    else:
        res, y_errors = Other_data(resOT)
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    x_ax = np.arange(len(res))
    ax.errorbar(x_ax,res,yerr=y_errors, capsize=8,linestyle='--' )
    plt.fill_between(x_ax,res+y_errors, res-y_errors, alpha=0.3)

    ax.set_xlabel('Temporal Distance of Prediction')
    ax.set_ylabel('Prediction Accuracy')

    folder = "../figs"
    name = "OneMemory"
    folder = utils.checkFolder(folder)
    fullname = utils.testTheName(folder + name, "pdf")
    plt.savefig(fullname)
    utils.plotMessage(fullname)
    plt.show()
    plt.close(fig)

def drw_overview_delay_1(resOT):
    def prepareData(resOT):
        res = np.mean(resOT, 0)
        idx = np.arange(len(res[0, :]))
        idx[0], idx[1] = 1, 0
        res = res[:, idx]
        y_errors = np.std(resOT, 0, dtype=np.float64)
        y_errors = y_errors[:, idx]
        dist = len(res[0]) - 2
        return res, y_errors, dist
    ### style of plotted vars ###
    # print(col)

    def prepareColor(res):
        for i in range(len(res)):
            if res[i][0] != res[i + 1][0]:
                length_line = i + 1
                amount_lines = int(len(res) / length_line)
                break

        cmap = mpl.cm.plasma
        col = cmap(np.linspace(0, .9, amount_lines))
        return col, length_line

    ### plot all lines ###
    def actualPlot_mean(ax, res, y_errors, length_line):
        cleg = []
        amount_of_lines = range(0, len(res), length_line)
        print(list(amount_of_lines))
        delay = 1
        j = delay
        for i in amount_of_lines:
            plo = np.transpose(res[i:i + length_line])
            pl_err = np.transpose(y_errors[i:i + length_line])
            if ax:
                ax.errorbar(plo[1], plo[j + 2], yerr=pl_err[j + 2],
                            capsize=8, c=col[int(i / length_line)],
                            )
            else:
                plt.errorbar(plo[1], plo[j + 2], yerr=pl_err[j + 2],
                                capsize=8, c=col[int(i / length_line)],
                                )
            cleg.append(r"$E_E$"+": %3.2f" % (plo[0, 0]))
        return cleg

    def make_leg(fig, cleg):
        leg_elem = []
        for i in range(len(cleg)):
            line = mpl.lines.Line2D([0], [0], c=col[i], label=cleg[i])
            leg_elem.append(line)
        fig.legend(
            bbox_to_anchor=(.97, .97), 
            # fontsize='small', 
            handles=leg_elem, loc="upper right")

    def make_labels():
        plt.ylabel('true prediction rate')
        plt.xlabel(r"$J_{EE}$")


    ### Acutal Function Body ###
    res, y_errors, dist = prepareData(resOT)
    col, length_line = prepareColor(res)

    fig = plt.figure(tight_layout=True, figsize=(8, 6))
    cleg= actualPlot_mean(None, res, y_errors, length_line, 
                                )
    make_leg(fig, cleg)
    make_labels()

    folder = "../figs"
    name = "delayOne"
    folder = utils.checkFolder(folder)
    fullname = utils.testTheName(folder + name, "pdf")
    plt.savefig(fullname)
    utils.plotMessage(fullname)
    plt.show()
    plt.close(fig)

def rar():
    print(np.random.randint(10))

def hamming_analyse():
    diff = np.load("hamming3_control.npy")[0][2]
    control = np.load("hamming3_control.npy")[1][2]
    diff = np.delete(diff,3, 0)
    # diff = np.load("hamming3.npy")[0][:,:21]
    # control = np.load("hamming3.npy")[1][:,:21]
    # diff = np.delete(diff,1, 0)
    diff_mean = np.mean(diff,axis=0)
    # idx = np.argmin(np.mean(diff,axis=1))
    # diff_mean = diff[idx]
    diff_std = np.std(diff,axis=0)

    control_mean = np.mean(control,axis=0)
    control_std = np.std(control,axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    dur = diff_mean.shape[0]
    time = np.arange(dur)
    # ax.plot(diff_mean)

    diff_txt = "One flipped"
    control_txt = "Random Pairings"

    ax.errorbar(time,diff_mean,yerr=diff_std, label=diff_txt)
    ax.errorbar(time,control_mean,yerr=control_std, label=control_txt)

    ax.set_xlabel('Timesteps since start')
    ax.set_ylabel('Share of all Neurons')
    ax.set_xticks(np.linspace(0,dur-1,5))
    ax.legend()



    folder = "../figs"
    name = "hamming"
    title = "Hamming Distance"
    plt.title(title)
    folder = utils.checkFolder(folder)
    fullname = utils.testTheName(folder + name, "pdf")
    plt.savefig(fullname)
    utils.plotMessage(fullname)
    plt.show()
    plt.close(fig)


def one_vs_zero():
    data_array = np.load("ml3_data.npy",allow_pickle=True)
    for data in data_array[0]:
        print(data)
    naughts = np.array([row for row in data_array if row[0][0] <0.05])
    fulls = np.array([row for row in data_array if row[0][0] > 0.2])
    naughts_mean = np.mean(naughts, axis=0   )
    naughts_std = np.std(naughts, axis=0   )
    fulls_mean = np.mean(fulls, axis=0   )
    fulls_std = np.std(fulls, axis=0   )
    add_base = np.array([1,1])[:,None]
    print(add_base.shape)
    print(naughts_mean.shape)

    # naughts_mean = np.concatenate((add_base,naughts_mean),axis=1)
    naughts_mean = np.hstack((add_base,naughts_mean))
    naughts_std = np.hstack((add_base*0,naughts_std))
    fulls_mean = np.concatenate((add_base*0,fulls_mean),axis=1)
    fulls_std = np.concatenate((add_base*0,fulls_std),axis=1)
    all_mean = np.concatenate((naughts_mean, fulls_mean), axis=0)
    all_std = np.concatenate((naughts_std, fulls_std), axis=0)
    # all_mean = all_mean[:,1:]
    # all_std = all_std[:,1:]

    # all_mean = np.concatenate((naughts_mean, fulls_mean), axis=0)
    print(all_mean.shape)
    x_ax = np.arange(all_mean.shape[1])
    description = [
        "fully activated network, excitatory", "fully activated network, inhibitory",
        "dead network, excitatory", "dead network, inhibitory",
        ]
    cols = ['b','r','b','r']
    stil = ['-','-','--','--']
    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    for i in range(4):
        plt.errorbar(x_ax,all_mean[i],all_std[i], capsize=4, 
                    label=description[i], c=cols[i], linestyle=stil[i])
    plt.legend()
    ax.set_xlabel('Timesteps since start')
    ax.set_ylabel('Share of all Neurons')
    # ax.set_xticks(np.linspace(0,dur-1,5))
    folder = "../figs"
    name = "one_or_zero"
    folder = utils.checkFolder(folder)
    fullname = utils.testTheName(folder + name, "pdf")
    plt.savefig(fullname)
    utils.plotMessage(fullname)
    plt.show()
    plt.close(fig)


def prediction_analyse():
    ### Extract Numbers ###
    np.set_printoptions(precision=3, suppress= True)
    # res3 = np.load("test_the_machine_0129.npy")
    res3 = np.load("data/vary_jEE_extE_0320/prediction.npy")
    res3 = np.load("prediction_1_1.npy")
    # res3 = np.load("data/vary_jEE_extE_0210/prediction.npy")
    # res3 = np.load("data/vary_jEE_extE_0210/ratio.npy")
    # res3 = np.load("data/vary_jEE_extE_0207/array.npy")
    print(res3)
    drw_small_overview(res3, 0)
    # drw_overview_delay_1(res3, 0)
    # drw_overviewOT(res3, 0)
def pred_ratios():
    res3 = np.load("data/vary_jEE_extE_0320/ratio.npy")
    y_name = 'Internal / External Input Ratio'
    save_name = 'ratios'
    drw_overviewOT(res3, y_name, save_name)
def pred_internal():
    res3 = np.load("data/vary_jEE_extE_0320/activity.npy")
    y_name = "Cum. Internal Excitatory Input"
    save_name = 'internal'
    drw_overviewOT(res3, y_name, save_name)
def pred_ones():
    res3 = np.load("prediction_1_1.npy")
    drw_small_overview(res3, 0)
def pred_1d():
    res3 = np.load("data/vary_jEE_extE_0320/prediction.npy")
    drw_overview_delay_1(res3)

if __name__ == '__main__':
    # prediction_analyse()
    # hamming_analyse()
    # pred_ratios()
    # pred_internal()
    # pred_ones()
    # pred_1d()
    one_vs_zero()
    # print(np.array([[1,1]]).T.shape)
"""
prediction zu ratio 
"""