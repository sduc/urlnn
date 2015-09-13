###########################
# file: ica.py            #
# Author: Sebastien Duc   #
###########################
import mix
import numpy as np
import fastICA as f
import math as m
import matplotlib.pyplot as pl
from scipy.io.wavfile import write as wavwrite
import os
import sys

def square_cos_ica(g,dg,icaall = True):
    data = mix.mixsquarecos()
    if icaall:
        x,y,nbsteps = f.fastICA(data,g,dg)
        return x,y,nbsteps
    else:
        X = f.fastICAall(data,g,dg,resize=False)
        return X

def mixsounds_ica(g, dg):
    data = mix.mixsounds()
    X = f.fastICAall(data,g,dg)
    return X


# g and its derivative used for the algorithm
g1 = lambda x: m.tanh(x) 
dg1 = lambda x: 1 - m.tanh(x)**2 

g2 = lambda x: x*m.exp(-(x**2)/2)
dg2 = lambda x: -m.exp(-(x**2)/2)*(x**2 - 1) 

g3 = lambda y: y**3
dg3 = lambda y: 3*y**2


# sig is the signal and n is the number of samples to plot
def plot(sig,n):
    pl.plot(np.arange(n), sig[:n])
    pl.show()


def info_message():
    print("use python ica.py mixsquarecos   if you want to apply fastica on the mixed square cos")
    print("use python ica.py mixsounds       if you want to apply fastica on the mixed sounds")
    print("")
    print("To choose the g function use python ica.py [mixsquarcos|mixsounds] [tanh|exp|ycube]")
    print("by default g is tanh")

# set the function g. Default is tanh
def set_g(arg):
    g = g1
    dg = dg1
    if len(arg) > 2:
        if arg[2] == "exp":
            g = g2
            dg = dg2
        elif arg[2] == "ycube":
            g = g3
            dg = dg3
        elif arg[2] == "tanh":
            pass
        else:
            info_message()
    return g,dg
# ---------------------------------------------------------
#                MAIN
# ---------------------------------------------------------

if len(sys.argv) > 1:
    g,dg = set_g(sys.argv)

    #      Cos and square (part 1)
    if sys.argv[1] == 'mixsquarecos':
        sig1,sig2,n = square_cos_ica(g,dg)
        plot(sig1,200)
        plot(sig2,200)

    #      Sounds (part 2)
    elif sys.argv[1] == 'mixsounds':
        sigs = mixsounds_ica(g,dg)
        sigs = np.array(sigs,dtype = 'uint8')
        # write in a file
        if 'output' not in os.listdir('.'):
            os.mkdir('output')
        for i in range(sigs.shape[0]):
            wavwrite('output/unmixedsound'+`i`+'.wav',8000,sigs[i])

    # used to test the quality of the functions (part 1)
    elif sys.argv[1] == 'convergence':
        mean = 0
        N = 100
        N_n = N
        for i in range(N):
            s1,s2,n = square_cos_ica(g,dg)
            if n >= 0:
                mean += n
            #if algo diverged then don't count it
            else:
                N_n = N_n - 1
            print(mean)
        print("The mean of the number of steps untill convergence is " +`mean/N_n`)

    # Cos and square (part 2)
    elif sys.argv[1] == 'mixsquarecosall':
        sigs = square_cos_ica(g,dg,icaall=False)
        plot(sigs[0],200)
        plot(sigs[1],200)

    else:
        info_message()

else:
    info_message()

