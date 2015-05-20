#prior_generation.py
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import fmin
from scipy.stats import weibull_min
from scipy.stats import exponweib
import math
import matplotlib.pyplot as plt
import numpy as np

import os
os.chdir("/Users/johannesmauritzen/research/wind_invest_model/")
from wind_turbine import wind_turbine
#Get data from yr.no from nearest måling stasjon
#Andenes for Andmyren
#18 km from sight
#http://www.yr.no/sted/Norge/Nordland/And%C3%B8y/Andenes_m%C3%A5lestasjon/statistikk.html
#maws - monthly average wind speed
#smws - strongest monthly wind speed
andenes_maws = [6.6, 7.9, 8.6, 7.8, 7.1, 5.7, 5.8, 5.8, 5.0, 3.9,5.9, 5.6]
andenes_smws = [19.1, 23.0, 24.8, 17.2, 20.0, 24.5, 14.9, 17.2, 14.3, 13.5, 14.6, 16.5]

#Harstad målestasjon
#46 km from sight
#45 moh
harstad_maws = [2.1, 2.5, 2.3, 2.4, 2.0, 1.7, 1.9, 1.7, 1.9, 1.3, 2.4, 2.3]
harstad_smws = [10.6, 10.4, 14.0, 11.0, 15.5, 9.7, 7.0, 9.9, 7.4, 6.7, 9.8, 8.7]

#Sortland
#3 moh
sortland_maws = [3.3, 4.8, 4.7, 3.5, 3.3, 3.3, 3.2, 3.4, 3.8, 3.9, 5.1, 4.0]
sortland_smws = [11.8, 13.3, 22.5, 13.3, 12.3, 21.5, 11.8, 12.3, 9.7, 8.7, 15.9, 13.3]

d = {"andenes": andenes_maws, "harstad":harstad_maws, "sortland":sortland_maws}

andmyren_local_maws = pd.DataFrame(d)
andmyran_long = pd.melt(andmyren_local_maws)

def fit_weibul_from_means(p):
	"""
	Takes in vector of monthly mean wind speeds, mu, 
	and fits the best weibull distribution by least squares
	by way of weibull mean function:
	mu=lambda*gamma(1+1/k)
	"""
	mu=andmyran_long.value
	sq_errors=(mu - p[0] * math.gamma(1+1/p[1]))**2
	return(sum(sq_errors))


lmbd_hat, k_hat=fmin(fit_weibul_from_means, [2,2])

def weib(x,k,lmbd):
     return (k / lmbd) * (x / lmbd)**(k - 1) * np.exp(-(x / lmbd)**k)

x_wind=np.linspace(0,50,200)
prior_dist=[weib(x, k_hat, lmbd_hat*np.sqrt(3/2)) for x in x_wind]

fig, ax = plt.subplots()
ax.plot(x_wind, prior_dist)
plt.show()

#now sample from distribution,
# turn into wind data and see how much it costs
prior_winds=weibull_min.rvs(c=k_hat, scale=(lmbd_hat*np.sqrt(3/2)), size=8760)

fig, ax = plt.subplots()
ax.plot(x_wind, prior_dist)
ax.hist(prior_winds, normed=1, bins=100)
plt.show()

#convert prior_winds to power output
#Data from Vestas power curve chart V90 - 3.0MW - approximately
wind_speed=np.array([4,5,6,7,8,9,10, 11, 12, 13, 14, 15])
power_kw_v90=np.array([85, 200, 350, 590, 900, 1300, 1720, 2150, 2560, 2840, 2980, 3000])

#Create instance of a wind turbine
v90_turbine = wind_turbine(curve_speeds=wind_speed, power_points = power_kw_v90)
power_output = np.sum(v90_turbine(prior_winds))

yearly_power_output = []

for i in range(300):
	prior_winds=weibull_min.rvs(c=k_hat, scale=(lmbd_hat*np.sqrt(3/2)), size=8760)
	yearly_power_output.append(np.sum(v90_turbine(prior_winds)))

plt.hist(np.array(yearly_power_output)*.02, bins=30, normed=1)
plt.show()



