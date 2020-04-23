import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import sirt


rg = np.random.default_rng([ord(c) for c in 'S535 COVID-19 Forecasting'])

k = np.arange(5000)
groups = np.array(['S', 'I', 'R'])

# times = pd.read_csv('times.csv', parse_dates=True)
times = pd.date_range(start='2020-04-01', end='2020-04-30', freq='0.25D')
data_raw = pd.read_csv('../Database/master_state_data.csv', index_col=0)
locations = data_raw.index
pop = xr.DataArray(0, coords=[times, locations, k, groups], dims=['t', 'loc', 'k', 'group'])
initial = pop[{'t':0}]
initial.loc[{'group':'I'}] = xr.DataArray(data_raw['positive_04102020'], dims=['loc']).expand_dims({'k': k}, -1)
initial.loc[{'group':'R'}] = 0  # data_raw['recovered']
initial.loc[{'group':'S'}] = xr.DataArray(data_raw['Pop'] - data_raw['positive_04102020'], dims=['loc']).expand_dims({'k': k}, -1)
# transport = xr.DataArray(pd.read_csv('transport.csv', index_col=0), coords=[locations, locations], dims=['loc', 'outloc'])
# transport = xr.DataArray(0, coords=[locations, locations], dims=['loc', 'outloc'])
initMeanS = xr.DataArray(10000*rg.pareto(3, locations.shape), coords=[locations], dims=['loc'])
transport = sirt.genTransportMatrix(initMeanS, xr.DataArray(rg.pareto(1, locations.shape*2), coords=[locations, locations], dims=['loc', 'outloc']), dim='loc', outdim='outloc', rate=0.01)

# rates = xr.Dataset(pd.read_csv('rates.csv'))

ds = xr.Dataset({'pop': pop,
                 'transport': transport,
                 # 'iRate': rates['infection'],
                 # 'rRate': rates['recovery'],
                 # 'dRate': rates['death']})
                 'iRate': xr.DataArray(rg.lognormal(1, 0.1, locations.shape), dims=['loc']),
                 'rRate': xr.DataArray(rg.lognormal(0.01, 0.001, locations.shape), dims=['loc']),
                 'dRate': xr.DataArray(rg.beta(0.01, 0.99, locations.shape), dims=['loc'])})

for i in range(len(times)-1):
    sirt.step(ds, times[i], times[i+1], rg=rg)

totpop = ds.pop.sum(dim='loc')

# fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ax1 = plt.subplots()
fig.set_size_inches(5, 3)
ax1.plot(times, totpop.loc[{'group':'I'}], color='black', alpha=0.02)
fig.autofmt_xdate()
plt.ylabel('Cases')
# ax2.set_yscale('log')
# ax2.plot(times, totpop.loc[{'group':'I'}], color='black', alpha=0.03)
plt.savefig('example-trajectories.pdf', bbox_inches='tight')
plt.savefig('example-trajectories.svg', bbox_inches='tight')
