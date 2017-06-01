# Load necessary modules 
import numpy as np
import math
from sklearn import linear_model
from sklearn.externals import joblib

class append_s2:
	def __init__(self):
		"""A method for initializing variables that will be used later"""
		self.predicted_s2_train = []
		self.predicted_s2_test = []
  
	def load_set(self):
		"""A method for loading dataset"""
		### Load in dataset that contains s2 order parameter and features information for all RNAs except 2l3e ###
		training = np.loadtxt("15.0A_2l3e_train_updated.txt", usecols=range(1, 79))
		### Load in testing set which contains information for only 2l3e ###
		testing = np.loadtxt("15.0A_2l3e_test_updated.txt", usecols=range(1, 79))
		### Divide training and testing sets into s2 order parameters and features ###
		self.y_train = training[:, 0]  # s2 order parameters for all RNAs except 2l3e
		self.x_train = training[:, 1:]  # features for all RNAs except 2l3e
		self.y_test = testing[:, 0]  # s2 order parameters for 2l3e only
		self.x_test = testing[:, 1:]  # features for 2l3e only
		#print self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape
 
	def load_saved_model(self):
		"""A function for training the linear regression model and obtaining predicted s2 order parameters"""
		self.reg = linear_model.LinearRegression()
		self.reg.fit(self.x_train, self.y_train)
		saved_model = ["2l3e_LR_15.0A.pkl"]
		for i in saved_model:
			loaded_model = joblib.load(i)
			s2_train = self.reg.predict(self.x_train)
			for t in s2_train:
				self.predicted_s2_train.append(t)
			s2_test = self.reg.predict(self.x_test)
			for v in s2_test:
				self.predicted_s2_test.append(v)
                     
	def append_s2(self): 
		"""A function for appending the predicted s2 order parameters from the LR model as an additional feature of the RF model"""
		with open("15.0A_2l3e_append_train.txt", "r+") as f:
			newlines = ['{} {}\n'.format(old.rstrip('\n'), toadd) for old, toadd in zip(f, self.predicted_s2_train)]
			f.seek(0)
			f.writelines(newlines)
		with open("15.0A_2l3e_append_test.txt", "r+") as f:
  			newlines = ['{} {}\n'.format(old.rstrip('\n'), toadd) for old, toadd in zip(f, self.predicted_s2_test)]
			f.seek(0)
			f.writelines(newlines)
  

