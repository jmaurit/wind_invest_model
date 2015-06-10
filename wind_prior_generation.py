#prior_generation.py
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


class wind_prior_generation(object):
	"""
	inputs a pd.DataFrame of average wind data
	returns sample from wind prior
	"""
	def __init__(self, wind_data):
		self.wind_data = wind_data
		#self.wind_data_long =pd.melt(wind_data)

		#initialize fitting model
		self.alpha_hat=np.empty(3) #initialize alpha estimates
		self.sigma_hat=np.empty(3) #initialize sigma estimates
		for i in range(3):
			self.alpha_hat[i], self.sigma_hat[i] = fmin(self.weibul_sq_error, [2,2], args=(self.wind_data.iloc[:,i].tolist(),))

		#parameters for gamma distribution of alpha and sigma
		self.shape_alpha, self.location_alpha, self.scale_alpha = gamma.fit(self.alpha_hat, floc=0)
		self.shape_sigma, self.location_sigma, self.scale_sigma = gamma.fit(self.sigma_hat, floc=0)

# 	def __call__(self):
# 		m= """\
# 		Weibull with parameters:
# 		shape: 		{shape}
# 		scale:		{scale}
# 		"""
# 		return(dedent(m.format(shape=self.alpha_hat, scale=self.sigma_hat*np.sqrt(3/2)))
# )

	def weibul_sq_error(self, p, wind_series):
		"""
		Takes in vector of monthly mean wind speeds, mu, 
		and fits the best weibull distribution by least squares
		by way of weibull mean function:
		mu=lambda*gamma(1+1/k)
		"""
		mu=wind_series
		sq_errors=(mu - p[1] * math.gamma(1+1/p[0]))**2
		return(sum(sq_errors))

	# def create_wind_prior(self, min_speed=0, max_speed=50, num_bins=200):
	# 	x_wind=np.linspace(min_speed,max_speed,num_bins)
	# 	prior_dist=[self.weib(x, self.alpha_hat, self.sigma_hat*np.sqrt(3/2)) for x in x_wind]
	# 	return(np.array([x_wind, prior_dist]))

	def sample_from_prior(self, months=1):
		#now sample from distribution,
		#start by drawing alpha hats and sigma hats for the distribution
		alpha_hats = gamma.rvs(a=self.shape_alpha, 
			loc=0, scale=self.scale_alpha, size=months)
		sigma_hats = gamma.rvs(a=self.shape_sigma,
		 loc=0, scale=self.scale_sigma, size=months)
		gamma_params = [i for i in zip(alpha_hats, sigma_hats)]
		month_sims = [self.generate_winds(i) for i in gamma_params]
		return(month_sims) #returns a list of monthly simulated data

	def generate_winds(self, p):
		prior_winds=weibull_min.rvs(c=p[0], scale=(p[1]), size=720)
		return(prior_winds)








