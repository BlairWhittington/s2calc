# Load necessary modules 
import numpy as np
import math
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.externals import joblib

max_features = 1.0
min_samples_leaf = 5
min_samples_split = 5
n_estimators = 1000
	
class model_building:
	def __init__(self, r_cut):
	"""Initializing the class variables that will be used later"""
		self.r_cut = r_cut 
		
	def get_training_testing(self, y, x):
		"""A method for obtaining training and testing sets"""
		dataset = np.loadtxt("15.0.txt", usecols = range(1, 80))	
		y = dataset[:, 0]
		x = dataset[:, 1]
		y.shape, x.shape 
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state=0) 
		x_train.shape, y_train.shape
		x_test.shape, y_test.shape
		
	def train_LR(self):
		"""A method for training the linear regression model"""
		reg = linear_model.LinearRegression()
		reg.fit(x_train, y_train)
		rmse = math.sqrt(np.mean((reg.predict(x_test)-y_test)**2))
		print rmse
		explained_variance = reg.score(x_test, y_test) 
		print explained_variance
		print(reg.coef_)
		joblib.dump(reg, "LR_r_cut.pkl", compress=3)
		
	def train_RF(self):
		"""A method for training the random forest model"""
		reg = RandomForestRegressor(bootstrap=True,criterion='mse', max_depth=None, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,n_estimators=n_estimators, n_jobs=1, oob_score=True, random_state=None,verbose=0)
		reg.fit(x_train,y_train)
		rmse = math.sqrt(np.mean((reg.predict(x_test)-y_test)**2))
		print rmse
		# Explained Variance 
		explained_variance = reg.score(x_test, y_test) 
		print explained_variance
		# Feature Importances
		print(reg.feature_importances_)
		# Save model
		joblib.dump(reg, "RF_r_cut.pkl", compress=3)
		
	def train_ERT(self):
		"""A method for training the extra randomized tree model"""
		reg = ExtraTreesRegressor(bootstrap=True,criterion='mse', max_depth=None, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,n_estimators=n_estimators, n_jobs=1, oob_score=True, random_state=None,verbose=0)
		reg.fit(x_train,y_train)
		rmse = math.sqrt(np.mean((reg.predict(x_test)-y_test)**2))
		print rmse
		# Explained Variance
		explained_variance = reg.score(x_test, y_test) 
		print explained_variance 
		# Feature Importances
		print(reg.feature_importances_)
		# Save model 
		joblib.dump(reg, "ERT_r_cut.pkl", compress=3)
