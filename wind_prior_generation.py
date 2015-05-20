#prior_generation.py
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import fmin
from scipy.stats import weibull_min
from scipy.stats import exponweib
import math
import matplotlib.pyplot as plt
import numpy as np
from textwrap import dedent


import os
os.chdir("/Users/johannesmauritzen/research/wind_invest_model/")

from wind_turbine import wind_turbine


class wind_prior_generation(object):
	"""
	inputs a pd.DataFrame of average wind data
	returns sample from wind prior
	"""
	def __init__(self, wind_data):
		self.wind_data = wind_data
		self.wind_data_long =pd.melt(wind_data)

		#initialize fitting model
		self.lmbd_hat, self.k_hat=fmin(self.weibul_sq_error, [2,2])

	def __call__(self):
		m= """\
		Weibull with parameters:
		shape: 		{shape}
		scale:		{scale}
		"""
		return(dedent(m.format(shape=self.k_hat, scale=self.lmbd_hat*np.sqrt(3/2)))
)

	def weibul_sq_error(self, p):
		"""
		Takes in vector of monthly mean wind speeds, mu, 
		and fits the best weibull distribution by least squares
		by way of weibull mean function:
		mu=lambda*gamma(1+1/k)
		"""
		mu=self.wind_data_long.value
		sq_errors=(mu - p[0] * math.gamma(1+1/p[1]))**2
		return(sum(sq_errors))

	def weib(self, x,k,lmbd):
	     return (k / lmbd) * (x / lmbd)**(k - 1) * np.exp(-(x / lmbd)**k)

	def create_wind_prior(self, min_speed=0, max_speed=50, num_bins=200):
		x_wind=np.linspace(min_speed,max_speed,num_bins)
		prior_dist=[self.weib(x, self.k_hat, self.lmbd_hat*np.sqrt(3/2)) for x in x_wind]
		return(np.array([x_wind, prior_dist]))

	def sample_from_prior(self, years=1):
		#now sample from distribution,
		# turn into wind data and see how much it costs
		prior_winds=weibull_min.rvs(c=self.k_hat, scale=(self.lmbd_hat*np.sqrt(3/2)), size=8760)
		return(prior_winds)







