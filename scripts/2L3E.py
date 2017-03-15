# Load necessary modules 
import numpy as np
import math
from sklearn import linear_model
from sklearn.externals import joblib

### Load in training set which contains information for all RNAs but 2l3e ###	
training = np.loadtxt("10.0.train.txt", usecols = range(1, 80))	
### Load in testing set which contains information for only 2l3e ###
testing = np.loadtxt("10.0.2l3e.txt", usecols = range(1,80))
### Divide training and testing sets into s2 order parameters and features ###
y_train = training[:, 0]
x_train = training[:, 1:]
y_test = testing[:, 0]
x_test = testing[:, 1:]
y_train.shape, x_train.shape , y_test.shape, x_test.shape

"""A method for training the linear regression model"""
### Fit linear regression model ###
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
### Compute RMSE ###
print(math.sqrt(np.mean((reg.predict(x_test)-y_test)**2)))
### Compute explained variance ###
print(reg.score(x_test, y_test))
### Compute feature importances ###
print(reg.coef_)
### Predict s2 order parameters ###
print reg.predict(x_train)

joblib.dump(reg, "LR_r_cut.pkl", compress=3) 

