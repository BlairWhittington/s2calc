### Load data ###
python
import numpy as np
import math
dataset = np.loadtxt("10.0.All.txt", usecols = range(1, 80))
y = dataset[:, 0]
x = dataset[:, 1:]
y.shape
x.shape

### Randomly split data into training and testing sets ###
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state=0) 
x_train.shape, y_train.shape
x_test.shape, y_test.shape

### Linear Regression ###
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
### Regression Coefficient 
print(regr.coef_)
### Mean Square Error(RMSE)
math.sqrt(np.mean((regr.predict(x_test)-y_test)**2))
### Explained Variance
regr.score(x_test, y_test) 

### Plotting ###
from matplotlib import pyplot
from numpy import arange
import bisect 
# Assign the variable x 
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Assign the variable y 
y = [Either RMSE or EV data]
# Title the scatterplot 
pyplot.title('RMSE vs. Fraction of Training')
pyplot.title('Explained Variance vs. Fraction of Training')
# Plot the assigned variables as dots connected by lines
pyplot.plot(x,y, marker='o', linestyle='-', color='b')
# Label the x-axis 
pyplot.xlabel('Fraction of Training')
# Label the y-axis
pyplot.ylabel('RMSE')
pyplot.ylabel('Explained Variance')
# Show the plot 
pyplot.show()
        
joblib.dump(reg, "LR_60Train_40Test.pkl", compress=3)

        
### Constructing Table of Features and Coefficients ###
### Convert regression coefficients to a list ###
Regression_Coef = reg.feature_importances_
Regression_Coef.tolist()
Features = ["C1'", "C2", "C5", "C6", "C8", "N1", "N3", "Resnum", "ADE", "CYT", "GUA", "URA", "Stacking", "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Chi", "nu0", "nu1", "nu2", "nu3", "nu4", "hEngery_base_base", "hEnergy_base_back", "hEnergy_back_back", "Dis_1", "Dis_2", "Dis_3", "Dis_4", "Dis_5", "Dis_6", "Dis_7", "Dis_8", "Dis_9", "Dis_10", "Dis_11", "Dis_12", "Dis_13", "Dis_14", "Dis_15", "Dis_16", "Dis_17", "Dis_18", "Dis_19", "Dis_20", "Dis_21", "Dis_22", "Dis_23", "Dis_24", "Dis_25", "Dis_26", "Dis_27", "Dis_28", "Dis_29", "Dis_30", "Dis_31", "Dis_32", "Dis_33", "Dis_34", "Dis_35", "Dis_36", "Dis_37", "Dis_38", "Dis_39", "Dis_40", "Dis_41", "Dis_42", "Dis_43", "Dis_44", "Dis_45", "Dis_46", "Dis_47", "Dis_48", "Dis_49", "Dis_50"]
Features = ["C1'", "C2", "C5", "C6", "C8", "N1", "N3", "Resnum", "ADE", "CYT", "GUA", "URA", "Stacking", "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Chi", "nu0", "nu1", "nu2", "nu3", "nu4", "hEngery_base_base", "hEnergy_base_back", "hEnergy_back_back"]
Features_List = list(range(28))

for l, c, d in zip(Features_List, Features, Regression_Coef):
 	print l, c, d
 	
 	
import numpy as np
import matplotlib.pyplot as plt
n_groups = 78
#Regression_Coef = reg.coef_
Regression_Coef = reg.feature_importances_
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 1.0
opacity = 0.8
rects1 = plt.bar(index, Regression_Coef, bar_width, alpha=opacity, color='b')
plt.xlabel('Features')
plt.ylabel('Feature Importances')
plt.title('Feature Importances vs. Features')
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
plt.xticks(x, labels)
plt.show()


