# Load necessary modules 
import numpy as np
import math
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

max_features = 1.0
min_samples_leaf = 5
min_samples_split = 5
n_estimators = 1000

### Load in training set which contains information for all RNAs but 2l3e ###	
training = np.loadtxt("15.0.train.txt", usecols = range(1, 80))	
### Load in testing set which contains information for only 2l3e ###
testing = np.loadtxt("15.0.2l3e.txt", usecols = range(1,80))
### Divide training and testing sets into s2 order parameters and features ###
y_train = training[:, 0] #s2 order parameters for all RNAs expect 2l3e
x_train = training[:, 1:] #features for all RNAs except 2l3e
y_test = testing[:, 0] #s2 order parameters only for 2l3e
x_test = testing[:, 1:] #features only for 2l3e
y_train.shape, x_train.shape , y_test.shape, x_test.shape

"""A method for training the linear regression model"""
### Fit linear regression model ###
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
### Fit Random Forest Model ###
reg = RandomForestRegressor(bootstrap=True,criterion='mse', max_depth=None, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,n_estimators=n_estimators, n_jobs=1, oob_score=True, random_state=None,verbose=0)
reg.fit(x_train,y_train)
### Compute RMSE ###
print(math.sqrt(np.mean((reg.predict(x_test)-y_test)**2)))
### Compute explained variance ###
print(reg.score(x_test, y_test))
### Compute feature importances ###
print(reg.coef_)
print(reg.feature_importances_)
### Predict s2 order parameters ###
s2 = reg.predict(x_test) #predicted s2 values for 2l3e only  
print s2.tolist()
### Save predictive model ###
joblib.dump(reg, "LR_15.0A_2l3e.pkl", compress=3) 

### Plotting s2 Order Parameters vs Residue Index ###
#Could import txt.files with data, plot all 7 bond vectors and complete for 15A, 20A, and 25A
import 2l3e_("")_10A_predicted.txt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from numpy import arange
import bisect
x = resid_index
y1 = predicted_LR
y2 = predicted_RF
y3 = measured_s2
plt.ylim(0.5, 1.1)
plt.title('s2 Order Parameters vs. Residue Index')
plt.plot(x,y1, marker='o', linestyle='-', color='b')
plt.plot(x,y2, marker='o', linestyle='-', color='k')
plt.plot(x, y3, marker='o', linestyle='-', color='r')
plt.xlabel('Residue Index')
plt.ylabel('s2 Order Parameters')
red_patch = mpatches.Patch(color='red', label='Measured s2 Order Parameters for 2l3e')
blue_patch = mpatches.Patch(color='blue', label='Predicted s2 Order Parameters for 2l3e (LR)')
black_patch = mpatches.Patch(color='black', label='Predicted s2 Order Parameters for 2l3e (RF)')
plt.legend(handles=[red_patch, blue_patch, black_patch])
plt.show()

### Plotting change in s2 Order Parameters vs Residue Index ###
def get_s2_vs_resid_plot(self):
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
y = []
for i in x:
	y.append(0)
y1 = (measured - predicted_LR)
y2 = (measured - predicted_RF)
y3 = (predicted_LR - predicted_RF)
plt.ylim(-0.10, 0.4)
plt.title('Difference in s2 Order Parameters vs. Residue Index')
plt.plot(x,y1, marker='o', linestyle='-', color='b')
plt.plot(x, y2, marker='o', linestyle='-', color='k')
#plt.plot(x, y3, marker='o', linestyle='-', color='g')
plt.plot(x,y, marker='None', linestyle='-', color='r')
blue_line = mlines.Line2D([],[], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (LR)')
black_line = mlines.Line2D([],[], color='black', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (RF)')
#green_line = mlines.Line2D([],[], color='green', marker='o', markersize=8, label='Predicted (LR) - Predicted s2 Order Parameters (RF)')
plt.legend(handles=[blue_line, black_line])
plt.xlabel('Residue Index')
plt.ylabel('Difference in s2 Order Parameters')
plt.show()

def get_measured_vs_predicted_plot(self):
 """A method for plotting measured vs. predicted s2 order parameters for 2l3e system"""
 x = self.measured s2
 y = self.measured s2
 y1 = self.predicted s2
 # Title Plot
 plt.title('Measured vs. Predicted s2 Order Parameters for 2l3e')
 plt.plot(x, y, marker='None', linestyle='-', color='k') 
 # (C1', H1')
 plt.plot(x[0:35],y1[0:35], marker='o', linestyle='None', color='r', label="(C1', H1')")
 # (C2, H2)
 plt.plot(x[35:40],y1[35:40], marker='o', linestyle='None', color='k', label="(C2, H2)")
 # (C5, H5)
 plt.plot(x[40:60],y1[40:60], marker='o', linestyle='None', color='g', label="(C5, H5)")
 # (C6, H6)
 plt.plot(x[60:80],y1[60:80], marker='o', linestyle='None', color='b', label="(C6, H6)")
 # (C8, H8)
 plt.plot(x[80:95],y1[80:95], marker='o', linestyle='None', color='m', label="(C8, H8)")
 # (N1, H1)
 plt.plot(x[95:105],y1[95:105], marker='o', linestyle='None', color='y', label="(N1, H1)")
 # (N3, H3)
 plt.plot(x[105:113],y1[105:113], marker='o', linestyle='None', color='c', label="(N3, H3)")
 # Create Legend
 plt.legend(bbox_to_anchor=(0.05, 0.7), loc=6, borderaxespad=0.)
 # Label x-axis
 plt.xlabel('Measured 2l3e s2 Order Parameters')
 # Label y-axis
 plt.ylabel('Predicted 2l3e s2 Order Parameters')
 # Show Plot
 