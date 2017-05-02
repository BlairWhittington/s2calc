#import scipy
#scipy.stats.kendalltau(x, y, initial_lexsort=None, nan_policy='propagate')
#from scipy import stats
#x1 = []
#x2 = []
#tau, p_value =  stats.kendalltau(x1, x2)
#tau
#p-value 



import scipy.stats as stats
import predicted_model_output.txt
from predicted_model_output.txt import *

def kendalltau_LR():
	x1 = y1
	x2 = y3
	tau, p_value = stats.kendalltau(y1, y3)
	print tau 
	
kendalltau_LR()