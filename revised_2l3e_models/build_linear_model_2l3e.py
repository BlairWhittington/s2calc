# Load necessary modules
import numpy as np
import math
from sklearn import linear_model
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from numpy import arange
import bisect
import scipy.stats as stats

class build_linear_model:
    """A method of initializing lists that will be used later"""
    def __init__(self):
        self.all_measured_s2 = []
        self.all_resid = []
        self.coef_list = []
        self.all_predicted_s2 = []
        self.all_s2_diff = []
        self.feature_num = []

    def load_set(self):
        """A method for loading dataset"""
        ### Load in dataset that contains s2 order parameter and features information for all RNAs except 2l3e ###
        training = np.loadtxt("25.0A_2l3e_train_updated.txt", usecols=range(1, 79))
        ### Load in testing set which contains information for only 2l3e ###
        testing = np.loadtxt("25.0A_2l3e_test_updated.txt", usecols=range(1, 79))
        ### Divide training and testing sets into s2 order parameters and features ###
        self.y_train = training[:, 0]  # s2 order parameters for all RNAs except 2l3e
        self.x_train = training[:, 1:]  # features for all RNAs except 2l3e
        self.y_test = testing[:, 0]  # s2 order parameters for 2l3e only
        self.x_test = testing[:, 1:]  # features for 2l3e only
        # print self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape
        ### Obtain a list of resids ###
        resid = np.loadtxt("25.0A_2l3e_test_updated.txt", usecols=range(1,80))
        self.resid = resid[:,78]
        for v in self.resid:
            self.all_resid.append(v)
        ### Obtain a list of measured s2 order parameters ###
        for i in self.y_test:
            self.all_measured_s2.append(i)

    def linear_regression(self):
        """A method for training the linear regression model"""
        self.reg = linear_model.LinearRegression()
        self.reg.fit(self.x_train, self.y_train)
        ### Save Linear Model ###
        joblib.dump(self.reg, "2l3e_LR_25.0A.pkl", compress=3)

    def compute_rmse(self):
        """A method for computing RMSE"""
        print math.sqrt(np.mean((self.reg.predict(self.x_test) - self.y_test) ** 2))

    def compute_explained_variance(self):
        """A method for computing Explained Variance"""
        print self.reg.score(self.x_test, self.y_test)

    def get_coefficients(self):
        """A method for obtaining regression coefficients"""
        for l in self.reg.coef_:
            self.coef_list.append(l)

    def get_predicted_s2(self):
        """A method for obtaining predicted s2 order parameters"""
        s2 = self.reg.predict(self.x_test)
        for t in s2:
            self.all_predicted_s2.append(t)

    def get_s2_diff(self):
        """A method for obtaining difference between measured and predicted s2 order parameters"""
        s2_diff = np.subtract(self.all_measured_s2, self.all_predicted_s2)
        for j in s2_diff:
            self.all_s2_diff.append(j)
            
    def get_kendall_tau(self):
    	"""A method for computing kendall tau coefficient between measured and predicted s2 order parameters"""
    	tau, p_value = stats.kendalltau(self.all_measured_s2, self.all_predicted_s2) 
    	print tau, p_value

    def make_table(self):
    	"""A method for making a table of all residue indices, measured s2 order parameters, predicted s2 order parameters, 
    	difference between measured and predicted s2 order parameters, and correlation coefficients"""
        for v, i, t, j, l in zip(self.all_resid, self.all_measured_s2, self.all_predicted_s2, self.all_s2_diff, self.coef_list):
            print v, i, t, j, l

    def plot_features(self):
        """A method for plotting regression coefficients vs. features for Linear Regression Model"""
        n_groups = 77
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 1.0
        opacity = 0.8
        for c in range(0, 78):
            self.feature_num.append(c)
        x = self.feature_num
        y = self.coef_list
        rects1 = plt.bar(index, y, bar_width, alpha=opacity, color='b')
        plt.xticks(np.arange(min(x), max(x) + 1, 3.0))
        plt.ylim(-0.4, 0.4)
        plt.xlabel('Features')
        plt.ylabel('Regression Coefficients')
        plt.title('Regression Coefficients vs. Features (LR)')
        plt.show()

    def plot_s2_resid_C1(self):
        """"A method for plotting s2 Order Parameter vs Residue Index (C1', H1')"""
        x = self.all_resid[0:35]
        y1 = self.all_predicted_s2[0:35]
        y2 = self.all_measured_s2[0:35]
        plt.title('s2 Order Parameters vs. Residue Index')
        plt.plot(x,y1, marker='o', linestyle='-', color='b')
        plt.plot(x,y2, marker='o', linestyle='-', color='r')
        plt.xlabel('Residue Index')
        plt.ylabel('s2 Order Parameters')
    	plt.ylim(0.1, 1.2)
        red_patch = mpatches.Patch(color='blue', label='Predicted s2 Order Parameters for 2l3e (LR)')
        blue_patch = mpatches.Patch(color='red', label='Measured s2 Order Parameters for 2l3e')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

    def plot_s2_resid_C2(self):
        """""A method for plotting s2 Order Parameter vs. Residue Index (C2, H2)"""
        x = self.all_resid[35:40]
        y1 = self.all_predicted_s2[35:40]
        y2 = self.all_measured_s2[35:40]
        plt.title('s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        plt.plot(x, y2, marker='o', linestyle='-', color='r')
        plt.xlabel('Residue Index')
        plt.ylabel('s2 Order Parameters')
        plt.ylim(0.4, 1.0)
        red_patch = mpatches.Patch(color='blue', label='Predicted s2 Order Parameters for 2l3e (LR)')
        blue_patch = mpatches.Patch(color='red', label='Measured s2 Order Parameters for 2l3e')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

    def plot_s2_resid_C5(self):
        """""A method for plotting s2 Order Parameter vs. Residue Index (C5, H5)"""
        x = self.all_resid[40:60]
        y1 = self.all_predicted_s2[40:60]
        y2 = self.all_measured_s2[40:60]
        plt.title('s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        plt.plot(x, y2, marker='o', linestyle='-', color='r')
        plt.xlabel('Residue Index')
        plt.ylabel('s2 Order Parameters')
        plt.ylim(0.2, 1.2)
        red_patch = mpatches.Patch(color='blue', label='Predicted s2 Order Parameters for 2l3e (LR)')
        blue_patch = mpatches.Patch(color='red', label='Measured s2 Order Parameters for 2l3e')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

    def plot_s2_resid_C6(self):
        """""A method for plotting s2 Order Parameter vs. Residue Index (C6, H6)"""
        x = self.all_resid[60:80]
        y1 = self.all_predicted_s2[60:80]
        y2 = self.all_measured_s2[60:80]
        plt.title('s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        plt.plot(x, y2, marker='o', linestyle='-', color='r')
        plt.xlabel('Residue Index')
        plt.ylabel('s2 Order Parameters')
        plt.ylim(0.1, 1.2)
        red_patch = mpatches.Patch(color='blue', label='Predicted s2 Order Parameters for 2l3e (LR)')
        blue_patch = mpatches.Patch(color='red', label='Measured s2 Order Parameters for 2l3e')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

    def plot_s2_resid_C8(self):
        """""A method for plotting s2 Order Parameter vs. Residue Index (C8, H8)"""
        x = self.all_resid[80:95]
        y1 = self.all_predicted_s2[80:95]
        y2 = self.all_measured_s2[80:95]
        plt.title('s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        plt.plot(x, y2, marker='o', linestyle='-', color='r')
        plt.xlabel('Residue Index')
        plt.ylabel('s2 Order Parameters')
        plt.ylim(0.4, 1.1)
        red_patch = mpatches.Patch(color='blue', label='Predicted s2 Order Parameters for 2l3e (LR)')
        blue_patch = mpatches.Patch(color='red', label='Measured s2 Order Parameters for 2l3e')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

    def plot_s2_resid_N1(self):
        """""A method for plotting s2 Order Parameter vs. Residue Index (N1, H1)"""
        x = self.all_resid[95:105]
        y1 = self.all_predicted_s2[95:105]
        y2 = self.all_measured_s2[95:105]
        plt.title('s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        plt.plot(x, y2, marker='o', linestyle='-', color='r')
        plt.xlabel('Residue Index')
        plt.ylabel('s2 Order Parameters')
        plt.ylim(0.5, 1.2)
        red_patch = mpatches.Patch(color='blue', label='Predicted s2 Order Parameters for 2l3e (LR)')
        blue_patch = mpatches.Patch(color='red', label='Measured s2 Order Parameters for 2l3e')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

    def plot_s2_resid_N3(self):
        """""A method for plotting s2 Order Parameter vs. Residue Index (N3, H3)"""
        x = self.all_resid[105:113]
        y1 = self.all_predicted_s2[105:113]
        y2 = self.all_measured_s2[105:113]
        plt.title('s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        plt.plot(x, y2, marker='o', linestyle='-', color='r')
        plt.xlabel('Residue Index')
        plt.ylabel('s2 Order Parameters')
        plt.ylim(0.3, 1.2)
        red_patch = mpatches.Patch(color='blue', label='Predicted s2 Order Parameters for 2l3e (LR)')
        blue_patch = mpatches.Patch(color='red', label='Measured s2 Order Parameters for 2l3e')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

    def plot_s2_diff_resid_C1(self):
        """"Plotting difference in s2 Order Parameters (measured - predicted) vs. Residue Index (C1', H1')"""
        x = self.all_resid[0:35]
        y1 = self.all_s2_diff[0:35]
        plt.axhline(y=0, linewidth=1.5, color='r')
        plt.title('Difference in s2 Order Parameters vs. Residue Index')
        plt.plot(x,y1, marker='o', linestyle='-', color='b')
        blue_line = mlines.Line2D([],[], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (LR)')
        plt.legend(handles=[blue_line])
        plt.ylim(-0.3, 0.45)
        plt.xlabel('Residue Index')
        plt.ylabel('Difference in s2 Order Parameters')
        plt.show()

    def plot_s2_diff_resid_C2(self):
        """"Plotting difference in s2 Order Parameters (measured vs. predicted) vs. Residue Index (C2. H2)"""
        x = self.all_resid[35:40]
        y1 = self.all_s2_diff[35:40]
        plt.axhline(y=0, linewidth=1.5, color='r')
        plt.title('Difference in s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        blue_line = mlines.Line2D([], [], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (LR)')
        plt.legend(handles=[blue_line])
        plt.ylim(-0.5, 0.3)
        plt.xlabel('Residue Index')
        plt.ylabel('Difference in s2 Order Parameters')
        plt.show()

    def plot_s2_diff_resid_C5(self):
        """"Plotting difference in s2 Order Parameters (measured vs. predicted) vs. Residue Index (C5, H5)"""
        x = self.all_resid[40:60]
        y1 = self.all_s2_diff[40:60]
       	plt.axhline(y=0, linewidth=1.5, color='r')
        plt.title('Difference in s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        blue_line = mlines.Line2D([], [], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (LR)')
        plt.legend(handles=[blue_line])
        plt.ylim(-0.3, 0.4)
        plt.xlabel('Residue Index')
        plt.ylabel('Difference in s2 Order Parameters')
        plt.show()

    def plot_s2_diff_resid_C6(self):
        """"Plotting difference in s2 Order Parameters (measured vs. predicted) vs. Residue Index (C6, H6)"""
        x = self.all_resid[60:80]
        y1 = self.all_s2_diff[60:80]
        plt.axhline(y=0, linewidth=1.5, color='r')
        plt.title('Difference in s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        blue_line = mlines.Line2D([], [], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (LR)')
        plt.legend(handles=[blue_line])
        plt.ylim(-0.4, 0.4)
        plt.xlabel('Residue Index')
        plt.ylabel('Difference in s2 Order Parameters')
        plt.show()

    def plot_s2_diff_resid_C8(self):
        """"Plotting difference in s2 Order Parameters (measured vs. predicted) vs. Residue Index (C8, H8)"""
        x = self.all_resid[80:95]
        y1 = self.all_s2_diff[80:95]
        plt.axhline(y=0, linewidth=1.5, color='r')
        plt.title('Difference in s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        blue_line = mlines.Line2D([], [], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (LR)')
        plt.legend(handles=[blue_line])
        plt.ylim(-0.1, 0.4)
        plt.xlabel('Residue Index')
        plt.ylabel('Difference in s2 Order Parameters')
        plt.show()

    def plot_s2_diff_resid_N1(self):
        """"Plotting difference in s2 Order Parameters (measured vs. predicted) vs. Residue Index (N1, H1)"""
        x = self.all_resid[95:105]
        y1 = self.all_s2_diff[95:105]
        plt.axhline(y=0, linewidth=1.5, color='r')
        plt.title('Difference in s2 Order Parameters vs. Residue Index')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        blue_line = mlines.Line2D([], [], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (LR)')
        plt.legend(handles=[blue_line])
        plt.ylim(-0.05, 0.4)
        plt.xlabel('Residue Index')
        plt.ylabel('Difference in s2 Order Parameters')
        plt.show()

    def plot_s2_diff_resid_N3(self):
        """"Plotting difference in s2 Order Parameters (measured vs. predicted) vs. Residue Index (N3, H3)"""
        x = self.all_resid[105:113]
        y1 = self.all_s2_diff[105:113]
        plt.title('Difference in s2 Order Parameters vs. Residue Index')
        plt.axhline(y=0, linewidth=1.5, color='r')
        plt.plot(x, y1, marker='o', linestyle='-', color='b')
        blue_line = mlines.Line2D([], [], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (LR)')
        plt.legend(handles=[blue_line])
        plt.ylim(-0.2, 0.3)
        plt.xlabel('Residue Index')
        plt.ylabel('Difference in s2 Order Parameters')
        plt.show()

    def plot_measured_predicted_s2(self):
        """A method for plotting measured vs. predicted s2 order parameters for 2l3e Model (LR)"""
        x = self.all_measured_s2
        y = self.all_measured_s2
        y1 = self.all_predicted_s2
        # Title Plot
        plt.title('All Measured vs. All Predicted s2 Order Parameters for 2l3e (LR)')
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
        plt.show()
