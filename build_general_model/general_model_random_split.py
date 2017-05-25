### Load data ###
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
import math 
from matplotlib import pyplot
from numpy import arange
import bisect 

#def load_data():
	#"""A method for loading data"""
	#dataset = np.loadtxt("10.0.All.txt", usecols = range(1, 80))
	#y = dataset[:, 0] # s2 order parameters
	#x = dataset[:, 1:] # features 
	#y.shape, x.shape 
	# Randomly split data into training and testing sets #
	#size =[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	#for i in size:
		#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = i, random_state=0) 
		#x_train.shape, y_train.shape, x_test.shape, y_test.shape
		#reg = linear_model.LinearRegression()
		#reg.fit(x_train, y_train) # Train Linear Model 
		#print math.sqrt(np.mean((reg.predict(x_test)-y_test)**2)) # Root Mean Square Error (RMSE)
		#print reg.score(x_test, y_test) # Explained Variance 
		#print(reg.coef_).tolist() # Regression Coefficient 

def get_plot():
	# Assign the variable x 
	x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	# Assign the variable y 
	y = [0.012882732636381022, 0.013684331086531009, 0.014607173948093632, 0.01368404296208559, 0.014994839580929401, 0.015362451278551228, 0.014725740961830892, 0.014830562992841346, 0.014268170628856278]
	# Title the scatterplot 
	#pyplot.title('RMSE vs. Fraction of Training')
	pyplot.title('Explained Variance vs. Fraction of Training')
	# Plot the assigned variables as dots connected by lines
	pyplot.plot(x,y, marker='o', linestyle='-', color='b')
	# Label the x-axis 
	pyplot.xlabel('Fraction of Training')
	# Label the y-axis
	#pyplot.ylabel('RMSE')
	pyplot.ylabel('Explained Variance')
	# Show the plot 
	pyplot.show()


def main():
	#load_data()
	get_plot()
	
main()