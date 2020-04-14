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
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt


rg = np.random.default_rng([ord(c) for c in 'S535 COVID-19 Forecasting'])
# rg = np.random.default_rng()


# def multinomial(n, p, dim=None, axis=None, out=None, rg=rg):
def multinomial(n, p, dim, out=None, rg=rg):
    n = xr.DataArray(n)
    p = xr.DataArray(p)

    condn = n.broadcast_like(p).copy()
    if out is None:
        out = condn.copy()

    # if dim not in p.dims:
    #     if axis is not None:
    #         dim = p.dims[axis]
    #     else:
    #         dim = p.dims[-1]
    p = p/p.sum(dim=dim)

    condp = xr.zeros_like(p)
    condp[{dim:0}] = p[{dim:0}]
    condp[{dim:slice(1, None)}] = (p[{dim:slice(1, None)}]/(1-p.shift({dim: 1}, 0)[{dim:slice(1, None)}].cumsum(dim=dim)))
    condp = np.clip(condp.fillna(1), 0, 1)

    for i in range(condp.sizes[dim]):
        out[{dim:i}] = rg.binomial(condn[{dim:i}], condp[{dim:i}])
        condn[{dim:slice(i+1, None)}] -= out[{dim:i}]

    return out

def genTransportMatrixOLD(s, t0):
    """
    Generates a transport matrix with a given stationary state s, starting from t0.
    """
    so = np.repeat(s[..., None], s.shape[-1], axis=-1)

    def f(x):
        y = x*so/np.einsum('...ij,...jk->...ik', x, so)
        return so*y/np.einsum('...ij,...jk->...ik', so, y)
    return optimize.fixed_point(f, t0, xtol=2*np.finfo(float).eps, method='iteration')

def genTransportMatrix(s, m0, dim, outdim, rate=1):
    """
    Generates a normalized transport rate matrix with a given stationary state s, starting from m0.
    """
    incoords = m0.coords[dim]
    outcoords = m0.coords[outdim]
    so = s.expand_dims({outdim: outcoords}, -1).rename({dim:outdim, outdim:dim})

    def f(x):
        y = x*so/x.dot(so, dims=[outdim])
        # print(np.abs(y-x*so/np.einsum('...ij,...jk->...ik', x, so)).max())
        z = y*so/y.dot(so.rename({dim:outdim, outdim:dim}), dims=[dim])
        # print(np.abs(z-so*y/np.einsum('...ij,...jk->...ik', so, y)).max())
        return z

    # m = optimize.fixed_point(f, m0, xtol=2*np.finfo(float).eps, method='iteration') - (incoords == outcoords)
    old = m0.copy()
    m = f(m0)
    for i in range(1000):
        old[:] = m
        m[:] = f(m)
        err = np.abs(m-old).max()
        if err < 2*np.finfo(float).eps:
            break
    lm = m.copy(data=logm(m))
    ident = (incoords == outcoords)
    return -rate*len(incoords)*(lm/lm.dot(ident, dims=[dim, outdim]))

def step(ds, t0, t1):
    dt = (t1-t0)/np.timedelta64(1,'D')
    current = ds.loc[{'t':t0}]
    new = ds.loc[{'t':t1}]
    finTransport = current.transport.copy(data=expm(dt*current.transport.values))
    new.pop[:] = multinomial(current.pop.sum(dim='loc'), finTransport.dot(current.pop, dims=['loc']).rename({'outloc':'loc'}), dim='loc')
    nS = new.pop.loc[{'group':'S'}]
    nI = new.pop.loc[{'group':'I'}]
    nR = new.pop.loc[{'group':'R'}]
    infections = rg.binomial(nS, -np.expm1(-new.iRate*dt*nI/(nS+nI+nR)))
    recoveries = nI.copy(data=rg.binomial(*xr.broadcast(nI, -np.expm1(-new.rRate*dt))))
    deaths = rg.binomial(*xr.broadcast(recoveries, -np.expm1(-new.dRate*dt)))
    nS[:] = nS - infections
    nI[:] = nI + infections - recoveries
    nR[:] = nR + recoveries - deaths


if __name__ == '__main__':
    # Constructs an example xarray Dataset for the simulation
    times = pd.date_range('2000-01-01', periods=90+1)
    locations = np.array(['MA', 'VT', 'CA', 'TX'])
    k = np.arange(1000)
    groups = np.array(['S', 'I', 'R'])

    pop = xr.DataArray(0, coords=[times, locations, k, groups], dims=['t', 'loc', 'k', 'group'])
    initMeanS = xr.DataArray(10000*rg.pareto(3, locations.shape), coords=[locations], dims=['loc'])
    initial = pop[{'t':0}]
    initial.loc[{'group':'S'}] = rg.poisson(initMeanS.expand_dims({'k': k}, -1))
    initial.loc[{'group':'I'}] = multinomial(1, xr.ones_like(initial.loc[{'group':'I'}], int), 'loc')
    initial.loc[{'group':'R'}] = 0

    transport = genTransportMatrix(initMeanS, xr.DataArray(rg.pareto(1, locations.shape*2), coords=[locations, locations], dims=['loc', 'outloc']), dim='loc', outdim='outloc', rate=0.01)

    ds = xr.Dataset({'pop': pop,
                     'transport': transport,
                     'iRate': xr.DataArray(rg.lognormal(1, 0.1, locations.shape), dims=['loc']),
                     'rRate': xr.DataArray(rg.lognormal(0.01, 0.001, locations.shape), dims=['loc']),
                     'dRate': xr.DataArray(rg.lognormal(0.01, 0.001, locations.shape), dims=['loc'])})

    for i in range(len(times)-1):
        step(ds, times[i], times[i+1])

    for l in locations:
        plt.plot(times, ds.pop.loc[{'loc':l, 'group':'I', 'k':0}])
    plt.show()
