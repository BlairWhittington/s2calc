# Load necessary modules 
import numpy as np
import math
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.externals import joblib

### Define variables to be used for random forest and extra randomized tree functions ###
max_features = 1.0
min_samples_leaf = 5
min_samples_split = 5
n_estimators = 1000

def load_set():
	"""A method for loading and splitting dataset"""
	### Load in dataset that contains s2 order parameter and features information for all RNAs ###
	dataset = np.loadtxt("10.0.All.txt", usecols = range(1, 80))	
	### Set y to s2 order parameters ###
	y = dataset[:, 0]
	### Set x to features ###
	x = dataset[:, 1:]
	### Define x_train, x_test, y_train, y_test as global variables ###
	global x_train
	global x_test
	global y_train
	global y_test
	### Randomly split dataset into training and testing ###
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state=0) 
	print x_train.shape, y_train.shape, x_test.shape, y_test.shape
		
def linear_regression():
	"""A method for training the linear regression model"""
	reg = linear_model.LinearRegression()
	reg.fit(x_train, y_train)
	### Compute RMSE ###
	#print math.sqrt(np.mean((reg.predict(x_test)-y_test)**2))
	### Compute Explained Variance ###
	#print reg.score(x_test, y_test) 
	### Compute Correlation Coefficient ###
	#print (reg.coef_)
	s2 = reg.predict(x_test) 
	print s2.tolist()
	### Save Linear Model ###
	#joblib.dump(reg, "LR_10.0A.pkl", compress=3)
		
def random_forest():
	"""A method for training the random forest model"""
	reg = RandomForestRegressor(bootstrap=True,criterion='mse', max_depth=None, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,n_estimators=n_estimators, n_jobs=1, oob_score=True, random_state=None,verbose=0)
	reg.fit(x_train,y_train)
	### Compute RMSE ###
	#print math.sqrt(np.mean((reg.predict(x_test)-y_test)**2))
	### Compute Explained Variance ###
	#print reg.score(x_test, y_test) 
	### Compute Feature Importances ###
	#print(reg.feature_importances_)
	### Predict s2 order parameters ###
	s2 = reg.predict(x_test) 
	print s2.tolist()
	### Save Random Forest Model ###
	#joblib.dump(reg, "RF_10.0A.pkl", compress=3)
		
def extra_randomized_trees():
	"""A method for training the extra randomized tree model"""
	reg = ExtraTreesRegressor(bootstrap=True,criterion='mse', max_depth=None, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,n_estimators=n_estimators, n_jobs=1, oob_score=True, random_state=None,verbose=0)
	reg.fit(x_train,y_train)
	### Compute RMSE ###
	print math.sqrt(np.mean((reg.predict(x_test)-y_test)**2))
	### Compute Explained Variance ###
	#print reg.score(x_test, y_test)  
	### Compute Feature Importances ###
	#print(reg.feature_importances_)
	s2 = reg.predict(x_test) 
	print s2.tolist()
	### Save Extra Randomized Tree Model ###
	#joblib.dump(reg, "ERT_10.0A.pkl", compress=3)

def main():
	"""A function that calls these functions"""
	load_set()
	linear_regression()
	random_forest()
	extra_randomized_trees()
	
main()
