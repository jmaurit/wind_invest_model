#simple_invest_model.py

import pystan
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sb
import numpy as np
from numpy import random as rd
from scipy.stats import exponweib
os.chdir("/Users/johannesmauritzen/research/wind_invest_model/")

wind_invest_code = """
data {
	int<lower=0> N; //number of observations
	vector[N] w; //number of wind speed observations
}
parameters {
	real<lower=0> alpha; //wind speed constrained to be positive and normal 
	real<lower=0> sigma;
}
model {
	w ~ weibull(alpha, sigma);
}
"""



#Step one - prior distribution from wind speed data from metreological office


#Step two - 
#open data, An
trial_data=pd.read_csv("Andmyran_2000.csv")
w=trial_data.Andmyran_a1.tolist()
N=len(trial_data)

wind_invest_data = {'N': N ,
	'w': w
}

fit = pystan.stan(model_code=wind_invest_code, data=wind_invest_data,
                  iter=10000, chains=4)

# fit2 = pystan.stan(fit=fit, data=wind_invest_data, iter=10000, chains=4)
# print(fit)

fit.plot()
plt.show()

la=fit.extract(permuted=True)
alpha_hat = la['alpha'].copy()
sigma_hat = la['sigma'].copy()
log_posterior = la['lp__'].copy()

plt.hist(alpha_hat, bins=100)
plt.show()

means=[sigma_hat[i]*math.gamma(1+1/alpha_hat[i]) for i in range(len(sigma_hat))]

fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.hist(alpha_hat, bins=100)
ax2.hist(sigma_hat, bins=100)
ax3.hist(means, bins=100)
plt.show()

rd.weibull()

def weib(x,s,a):
     return (a / s) * (x / s)**(a - 1) * np.exp(-(x / s)**a)

x_wind=np.linspace(0,50,200)

fig, ax = plt.subplots()
for i in range(200):
	wind_dist=[weib(x, sigma_hat[i], alpha_hat[i]) for x in x_wind]
	ax.plot(x_wind, wind_dist, alpha=.1)
ax.hist(w,normed=1, bins=50)
plt.show()



