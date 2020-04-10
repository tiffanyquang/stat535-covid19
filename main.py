#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naïve SIR model with transport.

@author: Aubrey Wiederin
"""

import numpy as np
# import scipy as sp
import pandas as pd
import xarray as xr
from scipy import optimize
from scipy.linalg import expm
import matplotlib.pyplot as plt


# rg = np.random.default_rng([ord(c) for c in 'S535 COVID-19 Forecasting'])
rg = np.random.default_rng()


def multinomial(n, p, dim=None, axis=None, out=None, rg=rg):
    n = xr.DataArray(n)
    p = xr.DataArray(p)

    condn = n.broadcast_like(p).copy()
    if out is None:
        out = condn.copy()

    if dim not in p.dims:
        if axis is not None:
            dim = p.dims[axis]
        else:
            dim = p.dims[-1]
    p = p/p.sum(dim=dim)

    condp = xr.zeros_like(p)
    condp[{dim:0}] = p[{dim:0}]
    condp[{dim:slice(1, None)}] = (p[{dim:slice(1, None)}]/(1-p.shift({dim: 1}, 0)[{dim:slice(1, None)}].cumsum(dim=dim)))
    condp = np.clip(condp.fillna(1), 0, 1)

    for i in range(condp.sizes[dim]):
        out[{dim:i}] = rg.binomial(condn[{dim:i}], condp[{dim:i}])
        condn[{dim:slice(i+1, None)}] -= out[{dim:i}]

    return out

def genTransportMatrixOLD(s, m0, rate=1):
    """
    Generates a normalized transport rate matrix with a given stationary state s, starting from m0.
    """
    so = np.repeat(s[..., None], s.shape[-1], axis=-1)

    def f(x):
        y = x*so/np.einsum('...ij,...jk->...ik', x, so)
        return so*y/np.einsum('...ij,...jk->...ik', so, y)
    m = optimize.fixed_point(f, m0, xtol=2*np.finfo(float).eps, method='iteration') - np.identity(s.shape[-1])
    return -(m/np.einsum('...ii->...', m))*rate

def genTransportMatrix(s, m0, dim, outdim, rate=1):
    """
    Generates a normalized transport rate matrix with a given stationary state s, starting from m0.
    """
    incoords = m0.coords[dim]
    outcoords = m0.coords[outdim]
    so = s.expand_dims({outdim: outcoords}, -1)


    def f(x):
        y = x*so/x.dot(so, dims=[dim])
        return so*y/so.dot(y, dims=[outdim])

    # m = optimize.fixed_point(f, m0, xtol=2*np.finfo(float).eps, method='iteration') - (incoords == outcoords)
    m = m0.copy()
    for i in range(10000):
        old = m
        m = f(m)
        if np.abs(m-old).max() < 2*np.finfo(float).eps:
            break
    ident = (incoords == outcoords)
    m -= ident
    return -(m/m.dot(ident, dims=[dim, outdim]))*rate

def step(ds, t0, t1):
    current = ds.loc[{'t':t0}]
    new = ds.loc[{'t':t1}]
    finT = current.transport.copy(data=expm((t1-t0)*current.transport.values))
    new.pop[:] = multinomial(current.pop.sum(dim='loc'), finT.dot(current.pop, dims=['loc']).rename({'outloc':'loc'}), dim='loc')


if __name__ == '__main__':
    # Constructs an example xarray Dataset for the simulation
    times = pd.date_range('2000-01-01', periods=365+1)
    locations = np.array(['MA', 'VT', 'CA', 'TX'])
    k = np.arange(1000)
    groups = np.array(['S', 'I', 'R'])

    pop = xr.DataArray(0, coords=[times, locations, k, groups], dims=['t', 'loc', 'k', 'group'])
    initMeanS = xr.DataArray(10000*rg.pareto(3, locations.shape), coords=[locations], dims=['loc'])
    initial = pop[{'t':0}]
    initial.loc[{'group':'S'}] = rg.poisson(initMeanS.expand_dims({'k': k}, -1), initial.loc[{'group':'S'}].shape)
    initial.loc[{'group':'I'}] = multinomial(1, xr.DataArray(np.ones_like(initial.loc[{'group':'I'}], int)))
    initial.loc[{'group':'R'}] = 0

    transport = genTransportMatrix(initMeanS, xr.DataArray(rg.pareto(1, locations.shape*2), coords=[locations, locations], dims=['loc', 'outloc']), dim='loc', outdim='outloc')

    ds = xr.Dataset({'pop': pop,
                     'transport': transport,
                     'beta': xr.DataArray(rg.lognormal(0.1, 0.01, locations.shape), dims=['loc']),
                     'gamma': xr.DataArray(rg.lognormal(0.02, 0.001, locations.shape), dims=['loc'])})

    for i in range(len(times)-1):
        step(ds, times[i], times[i+1])

    for l in locations:
        plt.plot(times, ds.pop.loc[{'loc':l, 'group':'S', 'k':0}])
