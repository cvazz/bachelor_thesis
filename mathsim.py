

import random 
import numpy as np
import timeit
from datetime import datetime
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy import special
import scipy.integrate as integrate
import scipy.optimize as opti




def meanMakerInfty(jM, extM, mean0):
    meanE = (jM[1]*extM[0] - jM[0]*extM[1])/(jM[0]-jM[1])*mean0
    meanI = (extM[0] - extM[1])/(jM[0]-jM[1])*mean0
    return np.array([meanE, meanI])

def meanMakerFinite(jM,extM,threshM,uM,mean0, K):
    fiddle = 1
    i = 0
    meanI = (extM[i]*mean0 - (threshM[i] - fiddle*uM[i])/math.sqrt(K))/(jM[0]-jM[1])
    i = 1
    meanI -= (extM[i]*mean0 - (threshM[i] - fiddle*uM[i])/math.sqrt(K))/(jM[0]-jM[1])
    meanE = jM[0]*meanI - extM[0]*mean0 + (threshM[0] -uM[0])/math.sqrt(K)
    return np.array([meanE, meanI])


def alphaMaker(jM, meanM):
    alpha = []
    for i in range(0,2):
        alpha.append((meanM[0] + meanM[1]*(jM[i])**2) ) 
    return np.array(alpha)

def approxH(x):
    return (-math.sqrt(2*math.pi) *abs(x)*math.e**(-x**2/2))

def approxInvH(x):
    return (-math.sqrt(2 *abs(math.log(x))))
def calcK(extM, jM, meanM, alphaM, threshM, mean0):
    K=[]
    for k in range(2):
        print((float(threshM[k] + math.sqrt(alphaM[k])*approxInvH(meanM[k]))/(extM[k]*mean0+ meanM[k] - jM[k]*meanM[(k+1)%2]))**2)
    return np.array(K)

def qMaker(meanM):
    q = meanM**2
    return q

def betaMaker(jM,qM):
    beta=[]
    for k in range(2):
        beta.append(qM[0]+qM[1]*(jM[k])**2)
    return beta

def calcU(meanM,alphaM):
    return np.array([(-math.sqrt(2*alphaM[i] *abs(math.log(meanM[i])))) for i in range(len(meanM))])

def meanFloatInput(jM, meanM, mean0, K, extM, threshM):
    u = []
    for i in range(2):
        w = extM[i] * mean0 + meanM[i]
        wip = (extM[i] * mean0 + meanM[i] + jM[i]*meanM[round((1+i)%2)])*math.sqrt(K)-threshM[i]
        u.append(wip)
    return np.array(u)


def do3(sizeE, sizeI, extE, extI, jE, jI, threshE, threshI):
    mean0 = 0.1 #REWORK

    sizeM = np.array([sizeE, sizeI])
    sizeMax = sizeE + sizeI

    extM = np.array([extE, extI])
    threshM = np.array([threshE, threshI])

    jM = np.array([jE,jI])
    meanM = meanMakerInfty(jM, extM, mean0)
    alphaM = alphaMaker(jM,meanM)
    qM = qMaker(meanM)
    betaM = betaMaker(jM,qM)
    uM = calcU(meanM,alphaM)

    #print(betaM)
    #print(alphaM)

    K = 1000 
    K2 = 1e9
    meanMakerFinite(jM,extM,threshM,uM,mean0, K)

    #res = integrate.quad(lambda x: meanMakerInfty(jM,extM, x)[0], 0 ,1)
    #uM = meanFloatInput(jM, meanM, mean0, K, extM, threshM) 
    plot3(jM,extM,threshM, K)
    #plot4(jM,extM, threshM ,K)

def calcDenser(uM,alphaM,betaM,m,x):
    i = 0
    infunc = (-uM[i] + math.sqrt(betaM[i])*x)/(math.sqrt(alphaM[i]-betaM[i]))
    return math.e**(-x**2/2)/math.sqrt(2*math.pi) *(m- special.erfc(infunc)/2)

    
def calcDensity(jM, meanM, m, extM, threshM ,K):
    #uM = meanFloatInput(jM, meanM, m, K, extM, threshM) 
    qM = qMaker(meanM)
    alphaM = alphaMaker(jM,meanM)
    uM = calcU( meanM, alphaM) 
    betaM = betaMaker(jM,qM)
    return integrate.quad(lambda x: calcDenser(uM,alphaM,betaM,m,x), -10 ,10)

def plot4(jM,extM, threshM ,K):
    m0 = np.linspace(0.0001, 0.3)
    #mM = [meanMakerInfty(jM,extM,m) for m in m0]
    mM = np.array([0.1,0.7])

    dM = [calcDensity(jM, mM, m0[i], extM, threshM, K)[0] for i in range(len(m0))]
    mM = np.transpose(mM)
    fig, ax = plt.subplots()
    plt.plot(m0/.1,dM)
    ax.set(xlabel='m0/m_E', ylabel='Density',
        title='Fig. 4 ')
    #plt.legend(["tbd"#,"tbd"],loc='upper left')
    fig.savefig("figs/test_fig4.png")
    plt.show()



def plot3(jM, extM,threshM, K):
    m0 = np.linspace(0.0001, 0.3)
    mM = [meanMakerInfty(jM,extM,m) for m in m0]
    alphaM = [alphaMaker(jM,meanM) for meanM in mM]
    uM = np.array([calcU( mM[i], alphaM[i]) for i in range(len(mM))])

    maM = [meanMakerFinite(jM,extM,threshM,uM[i],m0[i], K) for i in range(len(m0))]
    #print(maM)
    mM = np.transpose(mM)
    fig, ax = plt.subplots()
    plt.plot(m0,maM)
    plt.plot(m0,mM[0])
    plt.plot(m0,mM[1])
    ax.set(xlabel='External Rate', ylabel='Network rates',
        title='Fig. 3 ')
    plt.legend(["Excitatory, K =1000", "Inhibitory, K =1000", 
                "Excitatory, eq. 4.3 approx (same as inhib)", "Inhibitory, eq. 4.3 approx",],loc='upper left')
    fig.savefig("figs/test_fig3.png")
    plt.show()
    #ax.grid()



def task3():
    sizeE   = 10000
    sizeI   = 10000
    extE    = 1
    extI    = 0.7
    jE      = 2
    jI      = 1.8
    threshE = 1
    threshI = 0.7

    do3(sizeE, sizeI, extE, extI, jE, jI, threshE, threshI)
def meanActMaker(sqK,meanM,jM,extM,mean0):
    active=[]
    active.append(sqK*(meanM[0]+extM[0]*mean0))
    active.append(sqK*(meanM[1]*jM[1]))
    return np.array(active)

def devActMaker(qM,jM):
    dev=[]
    dev.append(qM[0])
    dev.append(qM[1]*jM[0])
    return np.array(dev)

def actMaker(meanActive,varActive,sizeM):
    list=[]
    for i in range(2):
        list.append( np.random.normal(meanActive[i],varActive[i],sizeM[i]) )
    return np.array(list)

def do6(K,mean0, tau, sizeE, sizeI, extE, extI, jE, jI, threshE, threshI):
    sqK = np.sqrt(K)
    sizeE = 10
    sizeI = 10
    sizeM = np.array([sizeE, sizeI])
    sizeMax = sizeE + sizeI

    extM = np.array([extE, extI])
    threshM = np.array([threshE, threshI])

    jM = np.array([jE,jI])
    meanM = meanMakerInfty(jM, extM, mean0)
    qM = qMaker(meanM)
    alphaM = alphaMaker(jM,meanM)
    betaM = betaMaker(jM,qM)
    uM = calcU(meanM,alphaM)
    meanActive = meanActMaker(sqK,meanM,jM,extM,mean0)
    print(meanActive)
    varActive = devActMaker(qM,jM)
    print(varActive)
    mActive = actMaker(meanActive,varActive,sizeM)
    print (mActive)
        
def task6():
    K=1000
    mean0 = 0.04
    sizeE   = 10000
    sizeI   = 10000
    extE    = 1
    extI    = 0.7
    jE      = 2
    jI      = 1.8
    threshE = 1
    threshI = 0.7
    tau     = 0.9
    do6(K,mean0,tau, sizeE, sizeI, extE, extI, jE, jI, threshE, threshI)



def mM_func(m0, inf, printit):
    mean0 = 0.1
    extE    = 1
    extI    = 0.7
    jE      = 2
    jI      = 1.8
    threshE = 1
    threshI = 0.7

    extM    = [extE, extI]
    threshM = [threshE, threshI]
    jM      = [jE, jI]
    K       = 1000

    act = 0
    ota = 1 - act

    if printit:
        print("m0")
        print(m0)
    A_E=  (extE*jI - extI*jE)/(jE-jI)
    A_I=  (extE - extI)/(jE-jI)
    mE =  (extE*jI - extI*jE)/(jE-jI) * m0
    mI =  (extE - extI)/(jE-jI) *m0
    mM = [mE, mI]
    if inf:
        return np.array([mE, mI])
    if printit:
        print("mM")
        print(mM)
    alphaM = [mE+jM[i]**2*mI for i in range(2)]
    if printit:
        print("alphaM")
        print(alphaM)
    hM = [-2*(np.log(m)) for m in mM]
    if printit:
        print("hM")
        print(hM)
    cM = [(threshM[i] + np.sqrt(alphaM[i])*hM[i])/np.sqrt(K) for i in range(2)]
    if printit:
        print("cM")
        print(cM)
    adeE = (jE*cM[1] -jI*cM[0])/(jE-jI)
    adeI = (cM[1] - cM[0])/(jE-jI)
    adeM = [adeE, adeI]
    if printit:
        print("add")
        print(adeM)
    mE = mE - (jE*cM[1] -jI*cM[0])/(jE-jI)
    mI = mI - (cM[1] - cM[0])/(jE-jI)
    # mE = mE - 0.02 *A_E- 0.2*m0
    # mI = mI - 0.02 *A_I - 0.2*m0
    try:
        mI = mI if mI>0 else 0
        mE = mE if mE>0 else 0
    except ValueError:
        mI = [m_I if m_I>0 else 0 for m_I in mI]
        mE = [m_E if m_E>0 else 0 for m_E in mE]
    return np.array([mE, mI])

def simpE(mE):
    m0      = 0.1
    extE    = 1
    extI    = 0.7
    jE      = 2
    jI      = 1.8
def numsolve():

    # # Plot it

    m0 = .2
    print( mM_func(m0,0,1))
    m0 = np.linspace(.0, .3, 51)
    mM = mM_func(m0, 1, 0)
    labM = ['mE_inf',"mI_inf"]
    [plt.plot(m0,m, label=lab) for m,lab in zip(mM,labM)]

    mM = mM_func(m0, 0, 0)
    labM = ['mE_1000',"mI_1000"]
    [plt.plot(m0,m, label=lab) for m,lab in zip(mM,labM)]
    plt.xlabel("m0")
    plt.ylabel("expression value")
    plt.legend()
    plt.grid()
    plt.show()

    # Use the numerical solver to find the roots

    # print(mM_func(*tau_initial_guess))
    # tau_solution = opti.fsolve(mM_func , tau_initial_guess)
    # tau_solution = opti.fsolve(simpE , 1)
    # print(tau_solution)
    



def main():
    # task3()
    # task6()
    #testRoutine()
    numsolve()


if __name__ == "__main__":
    main()
