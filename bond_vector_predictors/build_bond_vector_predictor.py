# Load necessary modules
import numpy as np
import math
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.externals import joblib
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from numpy import arange
import bisect

class build_bond_vector_predictor:
	"""A method of initializing lists that will be used later"""
	def __init__(self):
		self.all_measured_s2 = []
		self.LR_coef_list = []
		self.RF_importances_list = []
		self.ET_importances_list = []
		self.all_LR_predicted_s2 = []
		self.all_RF_predicted_s2 = []
		self.all_ET_predicted_s2 = []
		self.all_LR_s2_diff = []
		self.all_RF_s2_diff = []
		self.all_ET_s2_diff = []
		self.feature_num = []
		
	def load_set(self):
		"""A method for loading dataset"""
		### Load in dataset that contains measured s2 order parameters and features for all RNAs ###
		dataset = np.loadtxt("25.0A_all_N3.txt", usecols=range(1, 79))
		y = dataset[:, 0] # measured s2 order parameters
		x = dataset[:, 1:] # all features
		### Randomly split dataset into training and testing sets ###
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.4, random_state=0)
		#print self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape
		### Obtain a list of all measured s2 order parameters ###
		for i in self.y_test:
			self.all_measured_s2.append(i)
			
	def linear_regression(self):
		"""A method for training the linear regression model"""
		self.LR_reg = linear_model.LinearRegression()
		self.LR_reg.fit(self.x_train, self.y_train)
		### Save Linear Model ###
		joblib.dump(self.LR_reg, "N3_LR_predictor_25.0A.pkl", compress=3)
		
	def random_forest(self):
		"""A method for training the random forest model"""
		self.RF_reg = RandomForestRegressor(bootstrap=True, criterion='mse', max_features=1.0, min_samples_leaf=5,
											min_samples_split=5, n_estimators=1000, max_depth=None, n_jobs=1,
											oob_score=True, random_state=None, verbose=0)
		self.RF_reg.fit(self.x_train, self.y_train)
		### Save Random Forest Model ###
		joblib.dump(self.RF_reg, "N3_RF_predictor_25.0A.pkl", compress=3)
		
	def extra_trees(self):
		"""A method for training and extra randomized tree model"""
		self.ET_reg = ExtraTreesRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features=1.0, 
										min_samples_leaf=5, min_samples_split=5, n_estimators=1000, n_jobs=1, oob_score=True,
										random_state=None, verbose=0)
		self.ET_reg.fit(self.x_train, self.y_train)
		### Save Extra Trees Model ###
		joblib.dump(self.ET_reg, "N3_ET_predictor_25.0A.pkl", compress=3)
		
	def compute_rmse(self):
		"""A method for computing RMSE"""
		print math.sqrt(np.mean((self.LR_reg.predict(self.x_test) - self.y_test) ** 2))
		print math.sqrt(np.mean((self.RF_reg.predict(self.x_test) - self.y_test) ** 2))
		print math.sqrt(np.mean((self.ET_reg.predict(self.x_test) - self.y_test) ** 2))
		
	def compute_explained_variance(self):
		"""A method for computing Explained Variance"""
		print self.LR_reg.score(self.x_test, self.y_test)
		print self.RF_reg.score(self.x_test, self.y_test)
		print self.ET_reg.score(self.x_test, self.y_test)
    	
	def get_features(self):
		"""A method for obtaining regression coefficients/ feature importances"""
		for l in self.LR_reg.coef_:
			self.LR_coef_list.append(l)
			
		for l in self.RF_reg.feature_importances_:
			self.RF_importances_list.append(l)
		
		for l in self.ET_reg.feature_importances_:
			self.ET_importances_list.append(l)
			
	def get_predicted_s2(self):
		"""A method for obtainining predicted s2 order parameters"""
		LR_s2 =  self.LR_reg.predict(self.x_test)
		for t in LR_s2:
			self.all_LR_predicted_s2.append(t)
			
		RF_s2 = self.RF_reg.predict(self.x_test)
		for t in RF_s2:
			self.all_RF_predicted_s2.append(t)
			
		ET_s2 =  self.ET_reg.predict(self.x_test)
		for t in ET_s2:
			self.all_ET_predicted_s2.append(t)
			
	def get_s2_diff(self):
		"""A method for obtaining difference between measured and predicted s2 order parameters"""
		LR_s2_diff = np.subtract(self.all_measured_s2, self.all_LR_predicted_s2)
		for j in LR_s2_diff:
			self.all_LR_s2_diff.append(j)
			
		RF_s2_diff = np.subtract(self.all_measured_s2, self.all_RF_predicted_s2)
		for j in RF_s2_diff:
			self.all_RF_s2_diff.append(j)
			
		ET_s2_diff = np.subtract(self.all_measured_s2, self.all_ET_predicted_s2)
		for j in ET_s2_diff:
			self.all_ET_s2_diff.append(j)
			
	def get_kendall_tau(self):
		"""A method for computing kendall tau coefficient between measured and predicted s2 order parameters for LR"""
		tau, p_value = stats.kendalltau(self.all_measured_s2, self.all_LR_predicted_s2) 
		print tau, p_value

		tau, p_value = stats.kendalltau(self.all_measured_s2, self.all_RF_predicted_s2) 
		print tau, p_value
			
		tau, p_value = stats.kendalltau(self.all_measured_s2, self.all_ET_predicted_s2)
		print tau, p_value
			
	def make_table(self):
		"""A method for making a table of all measured s2 order parameters, predicted s2 order parameters,
		difference in s2 order parameters, and regression coefficients for the linear regression model"""
		for a, b, c, d, e, f, g, h, i, j in zip(self.all_measured_s2, self.all_LR_predicted_s2, self.all_RF_predicted_s2,
		 self.all_ET_predicted_s2, self.all_LR_s2_diff, self.all_RF_s2_diff, self.all_ET_s2_diff, self.LR_coef_list,
		 self.RF_importances_list, self.ET_importances_list):
			out = "%s %s %s %s %s %s %s %s %s %s" %(a, b, c, d, e, f, g, h, i ,j)
			print out
			
	def plot_features_LR(self):
		"""A method for plotting regression coefficients vs. features for Linear Regression Model"""
		n_groups = 77
		fig, ax = plt.subplots()
		index = np.arange(n_groups)
		bar_width = 1.0
		opacity = 0.8
		for c in range(0, 78):
			self.feature_num.append(c)
		x = self.feature_num
		y = self.LR_coef_list
		rects1 = plt.bar(index, y, bar_width, alpha=opacity, color='b')
		plt.xticks(np.arange(min(x), max(x) + 1, 3.0))
		#plt.ylim(-1.8, 1.0)
		plt.xlabel('Features')
		plt.ylabel('Regression Coefficients')
		plt.title('Regression Coefficients vs. Features (LR)')
		plt.show()		
											
	def plot_features_RF(self):
		"""A method for plotting feature importances vs. features for Random Forest Model"""
		n_groups = 77
		fig, ax = plt.subplots()
		index = np.arange(n_groups)
		bar_width = 1.0
		opacity = 0.8
		for c in range(0, 78):
			self.feature_num.append(c)
		x = self.feature_num
		y = self.RF_importances_list
		rects1 = plt.bar(index, y, bar_width, alpha=opacity, color='b')
		plt.xticks(np.arange(min(x), max(x) + 1, 3.0))
		plt.ylim(0, 0.6)
		plt.xlabel('Features')
		plt.ylabel('Feature Importances')
		plt.title('Feature Importances vs. Features (RF)')
		plt.show()	
		
	def plot_features_ET(self):
		"""A method for plotting feature importances vs. features for Extra Trees Model"""
		n_groups = 77
		fig, ax = plt.subplots()
		index = np.arange(n_groups)
		bar_width = 1.0
		opacity = 0.8
		for c in range(0, 78):
			self.feature_num.append(c)
		x = self.feature_num
		y = self.ET_importances_list
		rects1 = plt.bar(index, y, bar_width, alpha=opacity, color='b')
		plt.xticks(np.arange(min(x), max(x) + 1, 3.0))
		plt.ylim(0, 0.6)
		plt.xlabel('Features')
		plt.ylabel('Feature Importances')
		plt.title('Feature Importances vs. Features (ET)')
		plt.show()	
		
