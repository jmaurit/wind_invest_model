#wind_prior_experiment.py

#Experiment with creating new prior

import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import fmin
from scipy.stats import weibull_min
from scipy.stats import exponweib
from scipy.stats import gamma
import math
import matplotlib.pyplot as plt
import numpy as np
from textwrap import dedent
from scipy.stats import gamma



import os
os.chdir("/Users/johannesmauritzen/research/wind_invest_model/")

from wind_turbine import wind_turbine

index =  ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
 'sep', 'oct', 'nov', 'dec']

month = [i+1 for i in range(12)]

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

avg_wind_speed_data = pd.DataFrame({'andenes':andenes_maws, 
	'harstad':harstad_maws, 'sortland':sortland_maws})

wind_data=avg_wind_speed_data

def weibul_sq_error(p, wind_series):
	mu=wind_series
	sq_errors=(mu - p[1] * math.gamma(1+1/p[0]))**2
	return(sum(sq_errors))

alpha_hat=np.empty(3)
sigma_hat=np.empty(3)
for i in range(wind_data.shape[1]):
	alpha_hat[i], sigma_hat[i] = fmin(weibul_sq_error, [2,2], args=(wind_data.iloc[:,i].tolist(),))

#now estimate k and theta from alphas and sigmas
shape_alpha, location_alpha, scale_alpha = gamma.fit(alpha_hat, floc=0)
shape_sigma, location_sigma, scale_sigma = gamma.fit(sigma_hat, floc=0)

#now, sample from gamma distributions to get alpha and sigma
#then sample from weibul(alpha_hat, gamma_hat) to get data

#show gamma distribution for alpha and gamma
xs = np.linspace(0,6,200)
pdf_alpha=gamma.pdf(xs, a=shape_alpha, loc=.8, scale=scale_alpha)
pdf_sigma=gamma.pdf(xs, a=shape_sigma, loc=0, scale=scale_sigma)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(xs, pdf_alpha, "b-")
ax1.set_xlabel(r"$\alpha$")
ax2.plot(xs, pdf_sigma, "r-")
ax2.set_xlabel(r"$\sigma$")
fig.tight_layout()
fig.set_size_inches(6, 8)
#fig.savefig("figures/priors.png")
plt.show()


#creating fake data by sampling

alpha_hats = gamma.rvs(a=shape_alpha, loc=0, scale=scale_alpha, size=1000)
sigma_hats = gamma.rvs(a=shape_sigma, loc=0, scale=scale_sigma, size=1000)
gamma_params = [i for i in zip(alpha_hats, sigma_hats)]


def generate_winds(params):
	prior_winds=weibull_min.rvs(c=params[0], scale=(params[1]), size=720)
	return(prior_winds)

params = gamma_params[0]
params = (.8,3)
test_month = generate_winds(params)
fig, ax = plt.subplots()
ax.hist(test_month)
plt.show()

month_sims = [generate_winds(i) for i in gamma_params]


#now generate power
wind_speed=np.array([4,5,6,7,8,9,10, 11, 12, 13, 14, 15])
power_kw_v90=np.array([85, 200, 350, 590, 900, 1300, 1720, 2150, 2560, 2840, 2980, 3000])

#Create instance of a wind turbine
v90_turbine = wind_turbine(curve_speeds=wind_speed, power_points = power_kw_v90)

month_power=[]

for i in month_sims:
	month_power.append(v90_turbine(i))

avg_power=[]
for i in month_power:
	avg_power.append(np.array(i).sum())

fig, ax = plt.subplots()
ax.hist(avg_power, bins=30, normed=1)
ax.set_xlabel()	
plt.show()








