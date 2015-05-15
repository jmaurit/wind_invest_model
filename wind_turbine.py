
from textwrap import dedent
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sb
import math
import os
from scipy.interpolate import interp1d

os.chdir("/Users/johannesmauritzen/research/wind_invest_model/")

class wind_turbine(object):

	def __init__(self, curve_speeds, power_points, min_speed=4, rated_wind=15, rated_power=3000, cut_out=25):
		self.curve_speeds=curve_speeds
		self.power_points=power_points
		self.min_speed = min_speed
		self.rated_wind = rated_wind
		self.rated_power = rated_power
		self.cut_out = cut_out

		#initialize power function
		self.power_function
		self.power_f=interp1d(self.curve_speeds, self.power_points, kind="cubic")
	def __repr__(self):
		return self.__str__()

	def __str__(self):
		m = """\
        Wind turbine instance:
          - Min Speed (m/s)					: {ms}
          - Rated Power (kW)				: {p}
          - Rated Power at Wind Speed (m/s) : {ws} 
          - Cut out: {co}
        """
		return(dedent(m.format(ms=self.min_speed, p=self.rated_power, ws=self.rated_wind, co=self.cut_out)))

	def __call__(self, wind_speeds):
		#Call directly randomly generates wind speed 
		#from weibull distribution
		return(self.power_function(wind_speeds))
		

	def power_function(self, wind_speeds):
		"""
		takes in an array of wind_speeds, and exports power output according 
		to power curve
		"""
		print(wind_speeds)
		power_kw =[]
		for w in wind_speeds:
			if w<self.min_speed:
				power_kw.append(0)
			elif w>=self.cut_out:
				power_kw.append(0)
			elif w>=self.rated_wind and w<self.cut_out:
				power_kw.append(self.rated_power)
			else:
				power_kw.append(float(self.power_f(w)))
		return(power_kw)