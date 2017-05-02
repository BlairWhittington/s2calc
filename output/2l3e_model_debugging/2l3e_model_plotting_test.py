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
plt.ylim(0.3, 1.1)
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
plt.ylim(-0.15, 0.35)
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
 