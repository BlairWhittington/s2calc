# Load necessary modules 
import numpy as np 
import math
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

### Define variables to be used for random forest ###
max_features = 1.0
min_samples_leaf = 5
min_samples_split = 5
n_estimators = 1000
  
def load_set():
	"""A function for loading data set"""
	### Load in training set which contains information for all RNAs but 2l3e ###	
	training = np.loadtxt("25.0.train.txt", usecols = range(1, 80))	
	### Load in testing set which contains information for only 2l3e ###
	testing = np.loadtxt("25.0.2l3e.txt", usecols = range(1,80))
	### Divide training and testing sets into s2 order parameters and features ###
	global y_train
	global x_train
	global y_test
	global x_test
	y_train = training[:, 0] #s2 order parameters for all RNAs expect 2l3e
	x_train = training[:, 1:] #features for all RNAs except 2l3e
	y_test = testing[:, 0] #s2 order parameters only for 2l3e
	x_test = testing[:, 1:] #features only for 2l3e
	print y_test.tolist()
	print y_train.shape, x_train.shape , y_test.shape, x_test.shape
	
def linear_regression():
	"""A function for training linear regression model"""
	reg = linear_model.LinearRegression()
	reg.fit(x_train, y_train)
	### Compute RMSE ###
	#print(math.sqrt(np.mean((reg.predict(x_test)-y_test)**2)))
	### Compute explained variance ###
	#print(reg.score(x_test, y_test))
	### Compute feature importances ###
	#print(reg.coef_) 
	### Predict s2 order parameters ###
	s2 = reg.predict(x_test) #predicted s2 values for 2l3e only  
	print s2.tolist()
	### Save predictive model ###
	#joblib.dump(reg, "LR_25.0A_2l3e.pkl", compress=3) 
	

def random_forest():
	"""A function for training random forest model"""
	### Fit Random Forest Model ###
	reg = RandomForestRegressor(bootstrap=True,criterion='mse', max_depth=None, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,n_estimators=n_estimators, n_jobs=1, oob_score=True, random_state=None,verbose=0)
	reg.fit(x_train,y_train)
	### Compute RMSE ###
	#print(math.sqrt(np.mean((reg.predict(x_test)-y_test)**2)))
	### Compute explained variance ###
	#print(reg.score(x_test, y_test)
	### Compute feature importances ###
	#print(reg.feature_importances_)
	### Predict s2 order parameters ###
	s2 = reg.predict(x_test) #predicted s2 values for 2l3e only  
	print s2.tolist()
	### Save predictive model ###
	#joblib.dump(reg, "RF_25.0A_2l3e.pkl", compress=3) 

#def main():
	#load_set()
	#linear_regression()
	#random_forest()
	
#main()

load_set()
