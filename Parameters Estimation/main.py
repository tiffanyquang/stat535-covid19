import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats

def format_sci(x, s, ndig=2):
    neg = int(x < 0)
    nzeros = -math.floor(math.log10(s))-1
    scale = 10**(nzeros+ndig)
    rs = str(math.ceil(s*scale))
    rx = str(int(round(x*scale)))
    rxi = rx[neg:-(nzeros+ndig)]
    rxf = rx[-(nzeros+ndig):]
    return '{}{}.{}({})'.format('-'*neg, rxi.rjust(1,'0'), rxf, rs)


abbr = pd.read_csv('states.csv', index_col=0)


dates = pd.date_range(start='2020-01-22', end='2020-04-22')

raw_population = pd.read_csv('../JHData/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv', usecols=['Country_Region', 'Province_State', 'Admin2', 'Population'])
population = abbr.merge(raw_population[(raw_population['Country_Region'] == 'US') & (~pd.notna(raw_population['Admin2'])) & pd.notna(raw_population['Province_State']) & pd.notna(raw_population['Population'])], left_on='Name', right_on='Province_State').set_index('State')['Population']

data_cols = dates.strftime('%#m/%#d/%y').tolist()
raw_confirmed = abbr.merge(pd.read_csv('../JHData/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv', usecols=['Province_State']+data_cols, parse_dates=True), left_on='Name', right_on='Province_State').set_index('State')[data_cols]
raw_deaths = abbr.merge(pd.read_csv('../JHData/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv', usecols=['Province_State']+data_cols, parse_dates=True), left_on='Name', right_on='Province_State').set_index('State')[data_cols]

confirmed = raw_confirmed.groupby(['State']).sum().T
deaths = raw_deaths.groupby(['State']).sum().T


death_fraction = 0.01
recovered = deaths*(1-death_fraction)
infected = confirmed - recovered
susceptible = population - confirmed

dC = confirmed.apply(np.gradient)

beta = dC*population/(susceptible*infected)
beta[beta == np.inf] = np.NaN

with np.errstate(divide='ignore', invalid='ignore'):
    beta_log = np.log(beta)
beta_log[~np.isfinite(beta_log)] = np.NaN
beta_log_stats = beta_log.agg(['mean','sem','std']).T
beta_log_stats.index.set_names('State', inplace=True)
beta_log_all_mean = np.nanmean(beta_log.values.flat)
beta_log_all_std = np.nanstd(beta_log.values.flat)
beta_log_all_sem = stats.sem(beta_log.values.flat, nan_policy='omit')


print('Estimate of log(beta):', format_sci(beta_log_all_mean, beta_log_all_sem))
print('Standard deviation:', beta_log_all_std)
print(beta_log_stats)
beta_log_stats.to_csv('out/beta_log.csv')

beta_log_valid = beta_log.values.flat
beta_log_valid = beta_log_valid[~np.isnan(beta_log_valid)]

fig, ax = plt.subplots()
fig.set_size_inches(5, 3)
ax.hist(beta_log_valid, bins=100, density=True)
x = np.linspace(beta_log_valid.min(), beta_log_valid.max(), 300)
y = np.exp(-(x-beta_log_all_mean)**2/(2*beta_log_all_std**2))/(np.sqrt(2*np.pi)*beta_log_all_std)
ax.plot(x, y)
ax.set_xlabel(r'$\log(\beta)$')
fig.savefig('out/beta_hist.pdf', bbox_inches='tight')
fig.savefig('out/beta_hist.png', bbox_inches='tight')
plt.close()

for s in beta_log:
    valid = beta_log[s][~np.isnan(beta_log[s])]
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 3)
    ax.hist(valid, bins=20, density=True)
    x = np.linspace(valid.min(), valid.max(), 300)
    y = np.exp(-(x-beta_log_stats['mean'].loc[s])**2/(2*beta_log_stats['std'].loc[s]**2))/(np.sqrt(2*np.pi)*beta_log_stats['std'].loc[s])
    ax.plot(x, y)
    ax.set_xlabel(r'$\log(\beta)$')
    fig.savefig('out/'+s+'_hist.pdf', bbox_inches='tight')
    fig.savefig('out/'+s+'_hist.png', bbox_inches='tight')
    plt.close()
