#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naïve SIR model with transport.

@author: Aubrey Wiederin
"""

import numpy as np
import xarray as xr
from scipy.linalg import expm, logm


def multinomial(n, p, dim, out=None, rg=None):
    if rg is None:
        rg = np.random.default_rng()

    if not isinstance(n, xr.DataArray):
        n = xr.DataArray(n)
    if not isinstance(p, xr.DataArray):
        p = xr.DataArray(p)


    condn = n.broadcast_like(p[{dim:0}]).copy()
    if out is None:
        out = xr.zeros_like(p, int)

    p = p/p.sum(dim=dim)

    condp = np.clip(p/(1-p.shift({dim: 1}, 0).cumsum(dim=dim)), 0, 1)
    np.clip(p[{dim:0}], 0, 1, out=condp[{dim:0}].data)
    condp.data[np.isnan(condp.data)] = 1

    tk = condp.sizes[dim]
    for i in range(tk - 1):
        out[{dim:i}] = rg.binomial(condn, condp[{dim:i}])
        condn.data -= out[{dim:i}].data
    out[{dim:tk-1}] = rg.binomial(condn, condp[{dim:tk-1}])
    return out

def genTransportMatrix(s, m0, dim, outdim, rate=1):
    """
    Generates a normalized transport rate matrix with a given stationary state s, starting from m0.
    """
    incoords = m0.coords[dim]
    outcoords = m0.coords[outdim]
    so = s.expand_dims({outdim: outcoords}, -1).rename({dim:outdim, outdim:dim})

    def f(x):
        y = x*so/x.dot(so, dims=[outdim])
        z = y*so/y.dot(so.rename({dim:outdim, outdim:dim}), dims=[dim])
        return z

    old = m0.copy()
    m = f(m0)
    for i in range(1000):
        old.data[:] = m.data
        m.data[:] = f(m).data
        if (np.abs(m-old) < 2*np.finfo(float).eps).data.all():
            break
    lm = m.copy(data=logm(m))
    ident = (incoords == outcoords)
    return -rate*len(incoords)*(lm/lm.dot(ident, dims=[dim, outdim]))

def step(ds, t0, t1, rg=None):
    if rg is None:
        rg = np.random.default_rng()

    dt = (t1-t0)/np.timedelta64(1,'D')
    current = ds.loc[{'t':t0}]
    new = ds.loc[{'t':t1}]
    finTransport = current.transport.copy(data=expm(dt*current.transport.values).real)
    new.pop[:] = multinomial(current.pop.sum(dim='loc'), finTransport.dot(current.pop, dims=['loc']).swap_dims({'outloc':'loc'}), dim='loc')
    nS = new.pop.loc[{'group':'S'}]
    nI = new.pop.loc[{'group':'I'}]
    nR = new.pop.loc[{'group':'R'}]
    infections = nS - rg.binomial(nS, np.exp(-new.iRate*dt*nI/(nS+nI+nR+1)))
    recoveries = nI - rg.binomial(*xr.broadcast(nI, np.exp(-new.rRate*dt)))
    deaths = rg.binomial(*xr.broadcast(recoveries, new.dRate))
    nS.data[:] = nS.data - infections.data
    nI.data[:] = nI.data + infections.data - recoveries.data
    nR.data[:] = nR.data + recoveries.data - deaths.data
