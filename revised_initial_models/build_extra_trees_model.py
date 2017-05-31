# Load necessary modules 
import numpy as np
import math
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.externals import joblib
import matplotlib.pyplot as plt

class build_extra_trees_model:

	def __init__(self):
		"""A method for initializing lists that will be used later"""
		self.all_measured_s2 = []
		self.importances_list = []
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
		for i in self.y_test:
			self.all_measured_s2.append(i)
		#print self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape

	def extra_trees(self):
		"""A method for training the extra randomized tree model"""
		self.reg = ExtraTreesRegressor(bootstrap=True,criterion='mse', max_depth=None, max_features=1.0, min_samples_leaf=5, min_samples_split=5,n_estimators=1000, n_jobs=1, oob_score=True, random_state=None,verbose=0)
		self.reg.fit(self.x_train,self.y_train)
		### Save Extra Trees Model ###
		joblib.dump(self.reg, "initial_ERT_25.0.pkl", compress=3)
		
	def compute_rmse(self):
		"""A method for computing RMSE"""
		print math.sqrt(np.mean((self.reg.predict(self.x_test)-self.y_test)**2))
		
	def compute_explained_variance(self):
		"""A method for computing Explained Variance"""
		print self.reg.score(self.x_test, self.y_test) 

	def get_feature_importances(self):
		"""A method for obtaining feature importances"""
		for l in self.reg.feature_importances_:
			self.importances_list.append(l)
						
	def get_predicted_s2(self):
		"""A method for obtaining predicted s2 order parameters"""
		s2 = self.reg.predict(self.x_test) 
		for t in s2:
			self.all_predicted_s2.append(t)
				
	def make_table(self):
		for i, t, l in zip(self.all_measured_s2, self.all_predicted_s2, self.importances_list):
			print i, t, l

	def plot_features(self):
		"""A method for plotting coefficients vs. features for Extra Trees Model"""
		n_groups =  77
		fig, ax = plt.subplots()
		index = np.arange(n_groups)
		bar_width = 1.0
		opacity = 0.8
		for c in range(0, 78):
			self.feature_num.append(c)
		x = self.feature_num
		y = self.importances_list
		rects1 = plt.bar(index, y, bar_width, alpha=opacity, color='b')
		plt.xticks(np.arange(min(x), max(x)+1, 3.0))
		plt.ylim(0, 0.4)
		plt.xlabel('Features')
		plt.ylabel('Feature Importances') 
		plt.title('Feature Importances vs. Features (ERT)')
		plt.show()
