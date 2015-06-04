#simple_invest_model.py

import pystan
import pickle # for saving Stan fitted object
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
from numpy import random as rd
from scipy.stats import exponweib
from scipy.stats import weibull_min
import random

import os
os.chdir("/Users/johannesmauritzen/research/wind_invest_model/")
from wind_turbine import wind_turbine
from wind_prior_generation import wind_prior_generation



#First Create Prior***********************************
#Get data from yr.no from nearest måling stasjon
#Andenes for Andmyren
#18 km from sight
#http://www.yr.no/sted/Norge/Nordland/And%C3%B8y/Andenes_m%C3%A5lestasjon/statistikk.html
#maws - monthly average wind speed
#smws - strongest monthly wind speed

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

# fig, ax = plt.subplots()
# ax.plot(avg_wind_speed_data.month, avg_wind_speed_data.andenes, "b-")
# ax.text(6,7, "Andenes")
# ax.plot(avg_wind_speed_data.month, avg_wind_speed_data.harstad, "r-")
# ax.text(2,5, "Harstad")
# ax.plot(avg_wind_speed_data.month, avg_wind_speed_data.sortland, "g-")
# ax.text(6,2, "Sortland")
# ax.set_ylabel("Avg. Monthly Wind Speed")
# ax.set_xlabel("Month")
# ax.set_xlim(0,13)
# fig.set_size_inches(10,6)
# #fig.savefig("figures/avg_wind_speed_data.png")
# plt.show()

andmyran_prior = wind_prior_generation(avg_wind_speed_data)


wind, prior = andmyran_prior.create_wind_prior()
wind_sample = andmyran_prior.sample_from_prior()


fig, ax = plt.subplots()
ax.plot(wind, prior)
ax.hist(wind_sample, normed=1, bins=100)
ax.set_xlim(0,60)
ax.set_ylabel("Probability Density")
ax.set_xlabel("Wind Speed")
fig.set_size_inches(10,6)
fig.savefig("figures/prior_distribution.png")
plt.show()

#check that data and distributions look similar
sample_actual = np.random.choice(andmyran_prior.wind_data_long.value, size=1000)
sample_simulated = [np.mean(andmyran_prior.sample_from_prior(years=1/12)) for i in range(1000)]

fix, ax = plt.subplots()
ax.hist(sample_actual)
plt.show()

#convert prior_winds to power output
#Data from Vestas power curve chart V90 - 3.0MW - approximately
wind_speed=np.array([4,5,6,7,8,9,10, 11, 12, 13, 14, 15])
power_kw_v90=np.array([85, 200, 350, 590, 900, 1300, 1720, 2150, 2560, 2840, 2980, 3000])

#Create instance of a wind turbine
v90_turbine = wind_turbine(curve_speeds=wind_speed, power_points = power_kw_v90)

wind=np.linspace(0,30,200)
output = v90_turbine(wind)

power_sample=v90_turbine(wind_sample)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(wind, output, "-")
ax1.set_xlabel("Wind Speed, m/s")
ax1.set_ylabel("Power Output, kW")
ax2.hist(power_sample, bins=50, normed=1)
ax2.set_xlabel("Power Output")
ax2.set_ylabel("Density")
fig.set_size_inches(8,8)
fig.savefig("figures/power_output.png")
plt.show()

power_output = np.sum(v90_turbine(wind_sample))

yearly_power_output = []
for i in range(500):
	wind_sample=andmyran_prior.sample_from_prior()
	power = v90_turbine(wind_sample)
	yearly_power_output.append(np.sum(power))

plt.hist(np.array(yearly_power_output)*.02, bins=30, normed=1)
plt.show()


#Various scenarios

#loss functions - returns expected loss from options
def loss_invest(kwh, I, c_oper,p_kwh):
	return(I - (p_kwh - c_oper)*kwh)

def loss_pass(kwh,I, c_oper,p_kwh,d):
	return(d*((p_kwh-c_oper)*kwh - I))

def loss_wait(kwh, I, c_oper, p_kwh, d, M):
	stage2_loss=0
	if loss_invest(kwh, I, c_oper,p_kwh) < loss_pass(kwh, I, c_oper,p_kwh, d)  and loss_invest(kwh, I, c_oper,p_kwh)<0:
		stage2_loss = loss_invest(kwh, I, c_oper,p_kwh)
	else:
		stage2_loss =  loss_pass(kwh, I, c_oper,p_kwh, d)
	return(stage2_loss + M)

#fixed parameters
I = 100000 # fixed investment cost
c_oper = .005 #operating cost of turbine
M = 0 #fixed cost of measurement/value of waiting
p_kwh = .03 #price per kwh
d = .90 #discount factor for opportunity cost

kwh_dist = yearly_power_output # distribution of yearly power output


dist_loss_invest = []
dist_loss_pass = []
dist_loss_wait = []

for kwh in kwh_dist:
	dist_loss_invest.append(loss_invest(kwh, I, c_oper, p_kwh))
	dist_loss_pass.append(loss_pass(kwh, I , c_oper, p_kwh, d))
	dist_loss_wait.append(loss_wait(kwh, I, c_oper, p_kwh, d, M))

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.hist(dist_loss_invest, bins=50, normed=1)
ax1.set_xlabel("Expected Losses, Invest")
ax2.hist(dist_loss_pass, bins=50, normed=1)
ax2.set_xlabel("Expected Losses, Pass")
ax3.hist(dist_loss_wait, bins=50, normed=1)
ax3.set_xlabel("Expected Losses, Wait")
fig.tight_layout()
fig.set_size_inches(6, 10)
fig.savefig("figures/losses.png")
plt.show()

exp_loss_invest = np.sum(dist_loss_invest)/len(dist_loss_invest)
exp_loss_pass = np.sum(dist_loss_pass)/len(dist_loss_pass)
exp_loss_wait = np.sum(dist_loss_wait)/len(dist_loss_wait)

print(exp_loss_invest, exp_loss_pass, exp_loss_wait)

value_of_waiting = exp_loss_pass-exp_loss_wait
print("value of waiting", value_of_waiting)

lmbd_hat = andmyran_prior.lmbd_hat
k_hat =andmyran_prior.k_hat


wind_invest_code = """
data {
	int<lower=0> N; //number of observations
	real alpha_hat; //prior mean of alpha
	real sigma_hat; //prior mean of sigma
	vector[N] w; //wind speed observations

}
parameters {
	real<lower=0> alpha; 
	real<lower=0> sigma; 
}
model {
	alpha ~ normal(alpha_hat,1); //prior on alpha
	sigma ~ normal(sigma_hat, 1); //prior on sigma
	w ~ weibull(alpha, sigma); //likelihood
}
"""


#Step one - prior distribution from wind speed data from metreological office
alpha_hat =andmyran_prior.k_hat #alpha
sigma_hat = andmyran_prior.lmbd_hat #lambda

#Step two - 
#open data, An
trial_data=pd.read_csv("Andmyran_2000.csv")
w=trial_data.Andmyran_a1.tolist()
N=len(trial_data)


wind_invest_data = {'N': N ,
	'w': w,
	'alpha_hat':alpha_hat,
	'sigma_hat':sigma_hat
}

wind_invest_model = pystan.StanModel(model_code=wind_invest_code)
wind_invest_fit = wind_invest_model.sampling(data=wind_invest_data)
#fit_priors = pystan.stan(model_code=wind_invest_code, data=wind_invest_data,
 #                 iter=10000, chains=4)

with open('wind_invest_model.pkl', 'wb') as f:
    pickle.dump(wind_invest_model, f)

# load it at some future point
with open('wind_invest_model.pkl', 'rb') as f:
    wind_invest_model = pickle.load(f)

# fit2 = pystan.stan(fit=fit, data=wind_invest_data, iter=10000, chains=4)
# print(fit)

wind_invest_fit.plot()
plt.show()

la=wind_invest_fit.extract(permuted=True)
alpha_post_dist = la['alpha'].copy()
sigma_post_dist = la['sigma'].copy()
#log_posterior = la['lp__'].copy()


means=[sigma_post_dist[i]*math.gamma(1+1/alpha_post_dist[i]) for i in range(len(sigma_post_dist))]

def weib(x,s,a):
     return (a / s) * (x / s)**(a - 1) * np.exp(-(x / s)**a)

x_wind=np.linspace(0,50,200)

fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.hist(alpha_post_dist, bins=100, normed=1)
ax1.set_xlabel(r'$\alpha$ posterior distribution')
ax2.hist(sigma_post_dist, bins=100, normed=1)
ax2.set_xlabel(r'$\sigma$ posterior distribution')
for i in range(1000):
	wind_dist=[weib(x, sigma_post_dist[i], alpha_post_dist[i]) for x in x_wind]
	ax3.plot(x_wind, wind_dist, alpha=.1)
ax3.hist(w, normed=1, bins=100)
ax3.set_xlim(0,30)
ax3.set_xlabel("Posterior weibul distribution of wind speeds")
fig.tight_layout()
fig.set_size_inches(6, 10)
fig.savefig("figures/posteriors.png")
plt.show()

post_yearly_power=[]
#generate
for i in range(100):
	sigma_post_sample = np.random.choice(sigma_post_dist, 8760)
	alpha_post_sample = np.random.choice(alpha_post_dist, 8760)
	posterior_wind_data = [float(weibull_min.rvs(c=alpha_post_sample[i], 
	scale=sigma_post_sample[i], size=1)) for i in range(len(alpha_post_sample))]
	post_power_dist = v90_turbine(posterior_wind_data)
	post_yearly_power.append(np.sum(post_power_dist))

post_kwh_dist = post_yearly_power
post_dist_invest=[]
post_dist_pass=[]

for kwh in post_kwh_dist:
	post_dist_invest.append(loss_invest(kwh, I, c_oper, p_kwh))
	post_dist_pass.append(loss_pass(kwh, I , c_oper, p_kwh, d))

fig, (ax1, ax2) =plt.subplots(2, sharex=True)
ax1.hist(post_dist_invest, normed=1)
ax1.set_xlabel("Posterior Expected Losses, Invest")
ax1.set_ylabel("Probability Distribution")
ax2.hist(post_dist_pass, normed=1)
ax2.set_xlabel("Posterior Expected Losses, Pass")
ax2.set_ylabel("Probability Distribution")
fig.tight_layout()
fig.set_size_inches(6, 6)
fig.savefig("figures/post_losses.png")
plt.show()


