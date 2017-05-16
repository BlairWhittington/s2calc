### Import Modules ###
import scipy
import scipy.stats as stats

#def kendalltau_LR():
"""A function for computing the kendall tau coefficient 
between measured and LR predicted s2 order parameters"""
	#tau, p_value = stats.kendalltau(y1, y3)
	#print tau, p_value
	
#def kendalltau_RF():
"""A function for computing kendall tau coefficient 
between measured and RF predicted s2 order parameters"""
	#tau, p_value = stats.kendalltau(y2, y3)
	#print tau, p_value
	
	

import numpy as np
import matplotlib.pyplot as plt

def kendall_plot():
	n_groups = 7
	kendalltau_LR = [0.48907563025210066, -0.39999999999999991, 0.61052631578947369, 0.57894736842105265, 0.48571428571428582, 0.55555555555555569, 0.6428571428571429]
	kendalltau_RF = [0.45882352941176452, -0.39999999999999991, 0.41052631578947368, 0.45263157894736844,  0.31428571428571439, 0.7333333333333335, 0.4285714285714286]
	
	fig, ax = plt.subplots()
	index =  np.arange(n_groups)
	bar_width = 0.15
	opacity = 0.8

	rects1 = plt.bar(index, kendalltau_LR, bar_width, alpha=opacity, color = 'b', label='Measured & LR Predicted s2 Order Parameters (25A)')
	rects2 =  plt.bar(index + bar_width, kendalltau_RF, bar_width, alpha=opacity, color='k', label='Measured & RF Predicted s2 Order Parameters (25A)')
	plt.axhline(y=0, linewidth=2, color='r')
	
	plt.xlabel('Bond Vector')
	plt.ylabel('Kendall Tau Coefficient')
	plt.ylim(-0.7, 1.1)
	plt.title('Kendall Tau Coefficient by Bond Vector (25A)')
	plt.xticks(index + bar_width, ("C1', H1'", "C2, H2", "C5, H5", "C6, H6", "C8, H8", "N1, H1", "N3, H3"))
	plt.legend()

	plt.show()
	
kendall_plot()

	
