### Load necessary modules ###
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import bisect

### Plotting s2 Order Parameters vs Residue Index ###
def get_s2_vs_resid_plot():
"""A function for plotting s2 order parameter vs. residue index"""
	# Define x and y variables 
	x =  resid index
	y1 = s2 predicted_LR
	y2 = s2 predicted_RF
	y3 = s2 measured
	# Set y-axis boundary
	plt.ylim(0.1, 1.3)
	# Plot lines
	plt.plot(x,y1, marker='o', linestyle='-', color='b') # s2 predicted LR
	plt.plot(x,y2, marker='o', linestyle='-', color='k') # s2 predicted RF
	plt.plot(x, y3, marker='o', linestyle='-', color='r') # s2 measured
	# Title plot and axes
	plt.title('s2 Order Parameters vs. Residue Index')
	plt.xlabel('Residue Index')
	plt.ylabel('s2 Order Parameters')
	# Define and plot legend 
	red_patch = mpatches.Patch(color='red', label='Measured s2 Order Parameters for 2l3e')
	blue_patch = mpatches.Patch(color='blue', label='Predicted s2 Order Parameters for 2l3e (LR)')
	black_patch = mpatches.Patch(color='black', label='Predicted s2 Order Parameters for 2l3e (RF)')
	plt.legend(handles=[red_patch, blue_patch, black_patch])
	# Show plot 
	plt.show()

### Plotting difference in s2 Order Parameters vs Residue Index ###
def get_s2diff_vs_resid_plot():
"""A function for plotting difference in s2 order parameters vs. residue index"""
	# Define x and y variables 
	x = resid index
	y1 = (measured - predicted_LR)
	y2 = (measured - predicted_RF)
	y3 = (predicted_LR - predicted_RF)
	# Set y-axis boundary 
	plt.ylim(-0.35, 0.45)
	# Plot lines 
	plt.axhline(y=0, linewidth=2, color='r')
	plt.plot(x,y1, marker='o', linestyle='-', color='b')
	plt.plot(x, y2, marker='o', linestyle='-', color='k')
	plt.plot(x,y, marker='None', linestyle='-', color='r')
	blue_line = mlines.Line2D([],[], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (LR)')
	black_line = mlines.Line2D([],[], color='black', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (RF)')
	plt.legend(handles=[blue_line, black_line])
	plt.title('Difference in s2 Order Parameters vs. Residue Index')
	plt.xlabel('Residue Index')
	plt.ylabel('Difference in s2 Order Parameters')
	# Show plot
	plt.show()

def get_measured_vs_predicted_plot(self):
 """A method for plotting measured vs. predicted s2 order parameters for 2l3e system"""
	x = measured s2
 	y = measured s2
 	y1 = predicted LR or RF s2
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
	plt.show()
 