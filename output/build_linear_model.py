# Load necessary modules 
import numpy as np
import math
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.externals import joblib
import matplotlib.pyplot as plt

class build_linear_model:
	"""A method of initializing lists that will be used later"""
	def __init__(self):
		self.all_measured_s2 = []
		self.coef_list = []
		self.all_predicted_s2 = []
		self.feature_num = []
		
	def load_set(self):
		"""A method for loading and splitting dataset"""
		### Load in dataset that contains s2 order parameter and features information for all RNAs ###
		dataset = np.loadtxt("25.0A_all_updated.txt", usecols = range(1, 79)) 
		y = dataset[:, 0] # measured s2 order parameters
		x = dataset[:, 1:] # all features, except for residue number
		### Randomly split dataset into training and testing ###
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.4, random_state=0) 
		### Obtain a list of measured s2 order parameters ###
		for i in self.y_test:
			self.all_measured_s2.append(i)
		#print self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape
		
	def linear_regression(self):
		"""A method for training the linear regression model"""
		self.reg = linear_model.LinearRegression()
		self.reg.fit(self.x_train, self.y_train)
		### Save Linear Model ###
		joblib.dump(self.reg, "initial_LR_25.0.pkl", compress=3)
		
	def compute_rmse(self):
		"""A method for computing RMSE"""
		print math.sqrt(np.mean((self.reg.predict(self.x_test)-self.y_test)**2))
		
	def compute_explained_variance(self):
		"""A method for computing Explained Variance"""
		print self.reg.score(self.x_test, self.y_test) 

	def get_coefficients(self):
		"""A method for obtaining regression coefficients"""
		for l in self.reg.coef_:
			self.coef_list.append(l)
			
	def get_predicted_s2(self):
		"""A method for obtaining predicted s2 order parameters"""
		s2 = self.reg.predict(self.x_test) 
		for t in s2:
			self.all_predicted_s2.append(t)

	def make_table(self):
		for i, t, l in zip(self.all_measured_s2, self.all_predicted_s2, self.coef_list):
			print i, t, l
			
	def plot_features(self):
		"""A method for plotting regression coefficients vs. features for Linear Regression Model"""
		n_groups =  77
		fig, ax = plt.subplots()
		index = np.arange(n_groups)
		bar_width = 1.0
		opacity = 0.8
		for c in range(0, 78):
			self.feature_num.append(c)
		x = self.feature_num
		y = self.coef_list
		rects1 = plt.bar(index, y, bar_width, alpha=opacity, color='b')
		plt.xticks(np.arange(min(x), max(x)+1, 3.0))
		plt.ylim(-0.4, 0.4)
		plt.xlabel('Features')
		plt.ylabel('Regression Coefficients') 
		plt.title('Regression Coefficients vs. Features (LR)')
		plt.show()

	