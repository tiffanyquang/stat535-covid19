#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naïve SIR model with transport.

@author: Aubrey Wiederin
"""

import numpy as np
# import scipy as sp
# import xarray as xr  # TODO: Convert to DataArrays?
from scipy import optimize
import matplotlib.pyplot as plt


def genTransportMatrix(s, t0):
    """
    Generates a transport matrix with a given stationary state s, starting from t0.
    """
    so = np.repeat(s[..., None], s.shape[-1], axis=-1)

    def f(x):
        y = x*so/np.einsum('...ij,...jk->...ik', x, so)
        return so*y/np.einsum('...ij,...jk->...ik', so, y)
    return optimize.fixed_point(f, t0, xtol=2*np.finfo(float).eps, method='iteration')

def normalizeTMatrix(t, r):
    """
    Normalizes the transport matrix to have average of r coming out of each vertex.
    """
    tid = np.identity(t.shape[-1])
    dt = t-tid
    return tid - r*t.shape[-1]*dt/np.trace(dt)

def multinomial_rvs(count, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * count must be an (n-1)-dimensional numpy array.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.

    Modified from https://stackoverflow.com/q/55818845.
    """
    count = count.copy()
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = rg.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out


# TODO: Make this a class, or drasticly parallelize for different parameters.
rg = np.random.default_rng(12345)
nvert = 10  # Number of cities/states/etc.
sus = (10000*rg.pareto(3, nvert)).astype(int)  # Initial susceptible populations
inf = np.zeros_like(sus)  # Initial infected populations
inf[0] = 1
# Very sensitive to these parameters
beta = 0.0001  # Rate of infection per interaction
gamma = 0.02  # Recovery/death rate

# TODO: Replace with a data-driven model.
trate = 0.1
transport = normalizeTMatrix(genTransportMatrix(sus, rg.pareto(1, (nvert, nvert))), trate)


def step(sus, inf):
    """
    Iterates the simulation.

    TODO: Implement over an arbitrary timestep? Matrix exponentials, etc.
    """
    tS = multinomial_rvs(sus, transport.T).sum(axis=0)  # Hopefully these are implemented correctly.
    tI = multinomial_rvs(inf, transport.T).sum(axis=0)
    bIS = rg.binomial(tS, 1-(1-beta)**np.sqrt(inf))  # This should be replaced by a researched result.
    gI = rg.binomial(tI + bIS, gamma)
    return (tS - bIS, tI + bIS - gI)


n = 1000  # TODO: Multiple runs
k = 1000  # Number of timesteps
si = np.empty((k,2)+sus.shape, int)
si[0] = sus, inf
for i in range(1, k):
    si[i] = step(*si[i-1])

for i in range(nvert):
    # plt.plot(si[:,0,i])  # Plot uninfected
    plt.plot(si[:,1,i])  # Plot infected
    # plt.plot(si.sum(axis=1)[:, i])  # Plot Total
# plt.yscale('log')
plt.show()
