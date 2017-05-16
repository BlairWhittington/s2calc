# Load necessary modules 
import numpy as np
import math
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.externals import joblib
import matplotlib.pyplot as plt

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
	joblib.dump(reg, "general_LR_10.0.pkl", compress=3)
	
		
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
	#joblib.dump(reg, "general_RF_10.0A.pkl", compress=3)
		
def extra_randomized_trees():
	"""A method for training the extra randomized tree model"""
	reg = ExtraTreesRegressor(bootstrap=True,criterion='mse', max_depth=None, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,n_estimators=n_estimators, n_jobs=1, oob_score=True, random_state=None,verbose=0)
	reg.fit(x_train,y_train)
	### Compute RMSE ###
	#print math.sqrt(np.mean((reg.predict(x_test)-y_test)**2))
	### Compute Explained Variance ###
	#print reg.score(x_test, y_test)  
	### Compute Feature Importances ###
	#print(reg.feature_importances_)
	s2 = reg.predict(x_test) 
	print s2.tolist()
	### Save Extra Randomized Tree Model ###
	#joblib.dump(reg, "general_ERT_10.0A.pkl", compress=3)
	
#def plot_regression_coefficients():
	"""A method for plotting regression coefficients vs. features for Linear Regression Model"""
	#n_groups =  78
	#fig, ax = plt.subplots()
	#index = np.arange(n_groups)
	#bar_width = 1.0
	#opacity = 0.8
	#rects1 = plt.bar(index, y, bar_width, alpha=opacity, color='b')
	#plt.xticks(np.arange(min(x), max(x)+1, 3.0))
	#plt.xlabel('Features')
	#plt.ylabel('Feature Importances) 
	#plt.ylim(-0.3, 0.3)
	#plt.title('Feature Importances vs. Features')
	#plt.show()

#def plot_feature_importances():
	"""A method for plotting feature importances vs. features for Random Forest and Extra Randomized Trees"""
	#n_groups =  78
	#fig, ax = plt.subplots()
	#index = np.arange(n_groups)
	#bar_width = 1.0
	#opacity = 0.8
	#rects1 = plt.bar(index, y, bar_width, alpha=opacity, color='b')
	#plt.xticks(np.arange(min(x), max(x)+1, 3.0))
	#plt.xlabel('Features')
	#plt.ylabel('Feature Importances) 
	#plt.ylim(-0.3, 0.3)
	#plt.title('Feature Importances vs. Features')
	#plt.show()
	
#Features = ["C1'", "C2", "C5", "C6", "C8", "N1", "N3", "Resnum", "ADE", "CYT", "GUA", "URA", "Stacking", "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Chi", "nu0", "nu1", "nu2", "nu3", "nu4", "hEngery_base_base", "hEnergy_base_back", "hEnergy_back_back", "Dis_1", "Dis_2", "Dis_3", "Dis_4", "Dis_5", "Dis_6", "Dis_7", "Dis_8", "Dis_9", "Dis_10", "Dis_11", "Dis_12", "Dis_13", "Dis_14", "Dis_15", "Dis_16", "Dis_17", "Dis_18", "Dis_19", "Dis_20", "Dis_21", "Dis_22", "Dis_23", "Dis_24", "Dis_25", "Dis_26", "Dis_27", "Dis_28", "Dis_29", "Dis_30", "Dis_31", "Dis_32", "Dis_33", "Dis_34", "Dis_35", "Dis_36", "Dis_37", "Dis_38", "Dis_39", "Dis_40", "Dis_41", "Dis_42", "Dis_43", "Dis_44", "Dis_45", "Dis_46", "Dis_47", "Dis_48", "Dis_49", "Dis_50"]
#heavy_atom = y[0:7]
#residue_index = y[7:8]
#residue_name = y[8:12]
#stacking = y[12:13]
#torsion_angles = y[13:25]
#hydrogen_bonding = y[25:28]
#distances = y[28:78]
	
def main():
	"""A function that calls these functions"""
	load_set()
	linear_regression()
	#random_forest()
	#extra_randomized_trees()
	
main()
