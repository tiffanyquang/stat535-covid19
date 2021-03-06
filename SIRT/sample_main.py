import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import sirt


rg = np.random.default_rng([ord(c) for c in 'S535 COVID-19 Forecasting'])

# Constructs an example xarray Dataset for the simulation
times = pd.date_range(start='2020-04-01', end='2020-04-30', freq='0.25D')
# locations = np.array(['MA', 'VT', 'CA', 'TX'])
locations = np.arange(50)
k = np.arange(5000)
groups = np.array(['S', 'I', 'R'])

pop = xr.DataArray(0, coords=[times, locations, k, groups], dims=['t', 'loc', 'k', 'group'])
initMeanS = xr.DataArray(10000*rg.pareto(3, locations.shape), coords=[locations], dims=['loc'])
initial = pop[{'t':0}]
initial.loc[{'group':'S'}] = rg.poisson(initMeanS.expand_dims({'k': k}, -1))
# initial.loc[{'group':'I'}] = multinomial(1, xr.ones_like(initial.loc[{'group':'I'}], int), 'loc').broadcast_like(initial.loc[{'group':'I'}])
initial.loc[{'group':'I'}][{'loc':0}] = 1
initial.loc[{'group':'R'}] = 0

transport = sirt.genTransportMatrix(initMeanS, xr.DataArray(rg.pareto(1, locations.shape*2), coords=[locations, locations], dims=['loc', 'outloc']), dim='loc', outdim='outloc', rate=0.01)

ds = xr.Dataset({'pop': pop,
                 'transport': transport,
                 'iRate': xr.DataArray(rg.lognormal(1, 0.1, locations.shape), dims=['loc']),
                 'rRate': xr.DataArray(rg.lognormal(0.01, 0.001, locations.shape), dims=['loc']),
                 'dRate': xr.DataArray(rg.beta(0.01, 0.99, locations.shape), dims=['loc'])})

for i in range(len(times)-1):
    sirt.step(ds, times[i], times[i+1], rg=rg)

totpop = ds.pop.sum(dim='loc')

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(times, totpop.loc[{'group':'I'}], color='black', alpha=0.03)
ax2.set_yscale('log')
ax2.plot(times, totpop.loc[{'group':'I'}], color='black', alpha=0.03)
plt.show()
