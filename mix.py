"""This module provides the data and a whitening function for the 
mini-project 1 of the "Unsupervised and Reinforcement Learning class.

Usage:

    >>> import mix
    >>> data = mix.mixsounds()
    >>> data = whiten(data)
    >>> # etc...
"""

import numpy as np
from scipy.io.wavfile import read as wavread

def mixsounds():
    """Return 9 linear mixtures of sound signals.

    The sound signals have to be in '.../sources/'.
    """

    files = [('../sources/source%i.wav' % i) for i in range(1,10)]
    source = np.zeros((50000,9))
    for i in range(9):
        source[:,i] = wavread(files[i])[1]
    source -= np.mean(source, 0)
    mix = np.random.rand(9,9)
    data = np.dot(source, mix)
    return data

def mixsquarecos():
    """Return 2 linear mixtures of a square and a cos signals.
    """
    
    dim = 1001
    # generation of the 2 signals
    t = np.linspace(1,101,dim)
    
    # cos signal
    s1 = np.cos(2*t)
    s1 /= np.std(s1)
    
    # square signal
    s2 = np.ones(dim)
    s2[(np.arange(len(s2)) // 10) % 2 == 0] = -1
    s2 /= np.std(s2)
    
    s = np.c_[s1, s2]
    
    # mix the 2 signals
    A = np.random.rand(2,2) * 4
    x = np.dot(s,A)
    x /= np.std(x)
    
    return x

def whiten(data):
    """Return a whitened version of data.

    data should be a numpy array of size (n_samples, data_width).
    """
    C = np.cov(data.T)
    d, v = np.linalg.eig(C)
    wm = v / np.sqrt(d)
    return np.dot(data, wm)
