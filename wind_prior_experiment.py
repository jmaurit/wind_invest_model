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

month_means = wind_data.mean(axis=1)
month_sd = wind_data.std(axis=1)

location_means = wind_data.mean(axis=0)
location_sd = wind_data.std(axis=0)
series = 

alpha_hat=np.empty(3)
sigma_hat=np.empty(3)
for i in range(3):
	alpha_hat[i], sigma_hat[i] = fmin(weibul_sq_error, [2,2], args=(wind_data.iloc[:,i].tolist(),))

#now estimate k and theta from alphas and sigmas
shape_alpha, location_alpha, scale_alpha = gamma.fit(alpha_hat)
shape_sigma, location_sigma, scale_sigma = gamma.fit(sigma_hat)

def gamma_sq_error(p, series):
	mu=series
	sq_errors=(mu - )**2

def weibul_sq_error(p, wind_series):
	mu=wind_series
	sq_errors=(mu - p[1] * math.gamma(1+1/p[0]))**2
	return(sum(sq_errors))
