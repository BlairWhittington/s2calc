from sklearn.externals import joblib # joblib
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import math

# Load dataset
dataset = np.loadtxt("10.0.txt", usecols = range(1, 80))
dataset = np.loadtxt("15.0.txt", usecols = range(1, 80))
dataset = np.loadtxt("20.0.txt", usecols = range(1, 80))
dataset = np.loadtxt("25.0.txt", usecols = range(1, 80))
y = dataset[:, 0]
x = dataset[:, 1:]
y.shape, x.shape 
		
# Split into training and testing sets				
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state=0) 
x_train.shape, y_train.shape
x_test.shape, y_test.shape

# Linear Regression
reg = linear_model.LinearRegression()
# Train model
reg.fit(x_train, y_train)
# RMSE
math.sqrt(np.mean((reg.predict(x_test)-y_test)**2))
# Explained Variance
reg.score(x_test, y_test) 
# Feature Importances
print(reg.coef_)
# Save model
joblib.dump(reg, "LR_10.0.pkl", compress=3)
joblib.dump(reg, "LR_15.0.pkl", compress=3)
joblib.dump(reg, "LR_20.0.pkl", compress=3)
joblib.dump(reg, "LR_25.0.pkl", compress=3)

# Set parameters for RF 
max_features = 1.0
min_samples_leaf = 5
min_samples_split = 5
n_estimators = 1000

# RF
reg = RandomForestRegressor(bootstrap=True,criterion='mse', max_depth=None, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,n_estimators=n_estimators, n_jobs=1, oob_score=True, random_state=None,verbose=0)
# Train model
reg.fit(x_train,y_train)
# RMSE
math.sqrt(np.mean((reg.predict(x_test)-y_test)**2))
# Explained Variance 
reg.score(x_test, y_test) 
# Feature Importances
print(reg.feature_importances_)
# Save model
joblib.dump(reg, "RF_10.0.pkl", compress=3)
joblib.dump(reg, "RF_15.0.pkl", compress=3)
joblib.dump(reg, "RF_20.0.pkl", compress=3)
joblib.dump(reg, "RF_25.0.pkl", compress=3)

# Set parameters for ERT
max_features = 1.0
min_samples_leaf = 5
min_samples_split = 5
n_estimators = 1000
# Build ERT model 
reg = ExtraTreesRegressor(bootstrap=True,criterion='mse', max_depth=None, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,n_estimators=n_estimators, n_jobs=1, oob_score=True, random_state=None,verbose=0)
# Train model
reg.fit(x_train,y_train)
#RMSE
math.sqrt(np.mean((reg.predict(x_test)-y_test)**2))
# Explained Variance
reg.score(x_test, y_test) 
# Feature Importances
print(reg.feature_importances_)
# Save model 
joblib.dump(reg, "ERT_10.0.pkl", compress=3)
joblib.dump(reg, "ERT_15.0.pkl", compress=3)
joblib.dump(reg, "ERT_20.0.pkl", compress=3)
joblib.dump(reg, "ERT_25.0.pkl", compress=3)


