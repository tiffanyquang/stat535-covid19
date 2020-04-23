import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dates = pd.date_range(start='2020-01-22', end='2020-04-22')
# data_dir = '../JHData/csse_covid_19_data/csse_covid_19_daily_reports/'
# filenames = dates.strftime('%Y-%m-%d')
# pd_data = [pd.read_csv(data_dir+fn, usecols=['Country_Region', 'Province_State', 'Active', 'Deaths']) for fn in filenames]

data_cols = dates.strftime('%#m/%#d/%y').tolist()
raw_confirmed = pd.read_csv('../JHData/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv', usecols=['Province_State']+data_cols)
raw_deaths = pd.read_csv('../JHData/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv', usecols=['Province_State']+data_cols)


confirmed = raw_confirmed.groupby(['Province_State']).sum().T
deaths = raw_deaths.groupby(['Province_State']).sum().T

death_fraction = 0.01
recovered = deaths*(1-death_fraction)
infected = confirmed - recovered

fig, ax = plt.subplots()
fig.autofmt_xdate()
ax.plot(infected)
plt.show()
