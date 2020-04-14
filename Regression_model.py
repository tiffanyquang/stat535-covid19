#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:58:30 2020

@author: derrickliu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline

import statsmodels.formula.api as smf

from pandas import ExcelWriter
from pandas import ExcelFile

from sklearn import linear_model
import sklearn.model_selection as model_selection

covid_data = pd.read_csv('Database/master_state_data.csv')
covid_data = covid_data.dropna()

reg = linear_model.LinearRegression()
reg.fit(covid_data[['Density', 'LandArea', 'stateGDP', 'Airports', 
                    'Automobiles', 'Buses']], covid_data[['deaths_04102020']])

model = smf.ols(formula = 'deaths_04102020 ~ Density + LandArea + stateGDP + Airports + Automobiles + Buses', data = covid_data).fit()
summary = model.summary()
summary.tables[1]

model2 = smf.ols(formula = 'positive_04102020 ~ Density + LandArea + stateGDP + Airports + Automobiles + Buses', data = covid_data).fit()
summary = model2.summary()
summary.tables[1]