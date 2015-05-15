#power_curve.py
#last updated May 5th, 2015
#Translate wind power data or simulated wind power data into 
#power via power power_curve

#Power curve for V90 3.0 MW
#Cut in 3.5 MW
#Rated at 15
#cut out wind speed at 25
#cut in wind speed is below 20

#create power curve by regression of polynomial
#k + c_1v + c_2v^2+ c_3v^3
from textwrap import dedent
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sb
import math
from scipy.interpolate import interp1d

import os
os.chdir("/Users/johannesmauritzen/research/wind_invest_model/")
import wind_turbine

#Data from Vestas power curve chart V90 - 3.0MW - approximately
wind_speed=np.array([4,5,6,7,8,9,10, 11, 12, 13, 14, 15])
power_kw_v90=np.array([85, 200, 350, 590, 900, 1300, 1720, 2150, 2560, 2840, 2980, 3000])

power_f=interp1d(wind_speed, power_kw_v90, kind="cubic")

#Create instance of a wind turbine
v90_turbine = wind_turbine(curve_speeds=wind_speed, power_points = power_kw_v90)

#try with real data
wind_data=pd.read_csv("/Users/johannesmauritzen/research/wind_invest_model/wind_data.csv")
wind_data=wind_data.rename(columns = {'Unnamed: 0':'time'})
wind_data['time']=pd.to_datetime(wind_data["time"])
wind_data.set_index('time', inplace=True)
wind_data['year']=wind_data.index.year

#mean and standard deviation of wind data

#take a small section
Andmyran_2000_wind=wind_data.loc[wind_data.year==2000,'Andmyran_a1']
Andmyran_2000_wind=Andmyran_2000_wind.reset_index()
Andmyran_2000_wind.to_csv("Andmyran_2000.csv")
trial_data = np.array(Andmyran_2000_wind.Andmyran_a1)

power_output_v90=v90_turbine.power_function(trial_data)

#fit weibull distribution
from scipy import stats
p0, p1, p2  = stats.weibull_min.fit(trial_data)

x=np.linspace(0,45, 200)
weib_pdf=stats.weibull_min.pdf(x, p0, p1, p2)

#plot of raw wind speed data for a year
fig, ax = plt.subplots()
ax.hist(trial_data, 100, normed=True)
ax.plot(x, weib_pdf)
plt.show()

#plot of power data
fig, ax = plt.subplots()
ax.hist(power_curve_v90, 100, normed=True)
plt.show()















