import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import sirt
from scipy.linalg import logm


rg = np.random.default_rng([ord(c) for c in 'S535 COVID-19 Forecasting'])

k = np.arange(20000)
groups = np.array(['S', 'I', 'R'])
locations = np.array(['AK','AL','AR','AZ','CA','CO','CT','DE','FL','GA','HI','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY'], dtype=object)

times = pd.date_range(start='2020-04-01', end='2020-08-01', freq='0.25D')
data_raw = pd.read_csv('../Database/master_state_data.csv', index_col=1)
data_raw.index.set_names('loc', inplace=True)
data_raw = data_raw[data_raw.index.isin(locations)]

enplanements = pd.read_csv('enplanements.csv', index_col=0, squeeze=True)[locations]
enplanements.index.set_names('loc', inplace=True)
transport_rate = enplanements.divide(365*data_raw['Pop'], fill_value=0)
transport_rate += transport_rate.mean()/2
transport = xr.DataArray(transport_rate).expand_dims({'outloc':locations}).copy()
transport.data[transport['loc'] == transport['outloc']] = 0
transport /= transport.sum(dim='loc')/transport_rate
transport.data[transport['loc'] == transport['outloc']] = 1-transport_rate
transport.data = logm(transport)

pop = xr.DataArray(0, coords=[times, locations, k, groups], dims=['t', 'loc', 'k', 'group'])
initial = pop[{'t':0}]
initial.loc[{'group':'I'}] = xr.DataArray(data_raw['positive_04102020'], dims=['loc']).expand_dims({'k': k}, -1)
initial.loc[{'group':'R'}] = 0  # data_raw['recovered']
initial.loc[{'group':'S'}] = xr.DataArray(data_raw['Pop'] - data_raw['positive_04102020'], dims=['loc']).expand_dims({'k': k}, -1)


logbeta = pd.read_csv('../Parameters Estimation/out.csv', usecols=[0,1,3], index_col=0)
iRate = rg.lognormal(logbeta['mean'], logbeta['std'], size=k.shape+locations.shape*2)[..., 0].T

ds = xr.Dataset({'pop': pop,
                 'transport': transport,
                 'iRate': xr.DataArray(iRate, dims=['loc', 'k']),
                 'rRate': xr.DataArray(rg.lognormal(np.log(0.1), 0.1, locations.shape+k.shape), dims=['loc', 'k']),
                 'dRate': xr.DataArray(rg.beta(0.01, 0.99, locations.shape+k.shape), dims=['loc', 'k'])})

for i in range(len(times)-1):
    sirt.step(ds, times[i], times[i+1], rg=rg)

totpop = ds.pop.sum(dim='loc')

fig, ax = plt.subplots()
fig.set_size_inches(9, 5)
ax.plot(times, totpop.loc[{'group':'I'}], color='black', alpha=1/255, linewidth=0.5)
fig.autofmt_xdate()
plt.ylabel('Cases')
fig.savefig('infected.pdf', bbox_inches='tight')
fig.savefig('infected.png', bbox_inches='tight', dpi=72*4)
ax.set_yscale('log')
fig.savefig('infected.log.pdf', bbox_inches='tight')
fig.savefig('infected.log.png', bbox_inches='tight', dpi=72*4)
plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(9, 5)
ax.plot(times, (totpop.sum(dim='group').loc[{'t':times[0]}] - totpop.sum(dim='group')).T, color='black', alpha=1/255, linewidth=0.5)
fig.autofmt_xdate()
plt.ylabel('Cases')
fig.savefig('deaths.pdf', bbox_inches='tight')
fig.savefig('deaths.png', bbox_inches='tight', dpi=72*4)
ax.set_yscale('log')
fig.savefig('deaths.log.pdf', bbox_inches='tight')
fig.savefig('deaths.log.png', bbox_inches='tight', dpi=72*4)
plt.close()
