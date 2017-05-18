import numpy as np 
import math
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import matplotlib.pyplot as plt

max_features = 1.0
min_samples_leaf = 5
min_samples_split = 5
n_estimators = 1000


def load_set():
	training = np.loadtxt("10.0A_train.txt", usecols = range(1, 81))	
	### Load in testing set which contains information for only 2l3e ###
	testing = np.loadtxt("10.0A_test.txt", usecols = range(1,81))
	### Divide training and testing sets into s2 order parameters and features ###
	global y_train
	global x_train
	global y_test
	global x_test
	y_train = training[:, 0] #s2 order parameters for all RNAs expect 2l3e
	x_train = training[:, 1:] #features for all RNAs except 2l3e
	y_test = testing[:, 0] #s2 order parameters only for 2l3e
	x_test = testing[:, 1:] #features only for 2l3e
	y_train.shape, x_train.shape , y_test.shape, x_test.shape
	
		
def random_forest():
	"""A function for training random forest model"""
	### Fit Random Forest Model ###
	reg = RandomForestRegressor(bootstrap=True,criterion='mse', max_depth=None, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,n_estimators=n_estimators, n_jobs=1, oob_score=True, random_state=None,verbose=0)
	reg.fit(x_train,y_train)
	### Compute RMSE ###
	#print(math.sqrt(np.mean((reg.predict(x_test)-y_test)**2)))
	### Compute explained variance ###
	#print(reg.score(x_test, y_test))
	### Compute feature importances ###
	feature = reg.feature_importances_
	#print feature.tolist()
	### Predict s2 order parameters ###
	s2 = reg.predict(x_test) #predicted s2 values for 2l3e only  
	print s2.tolist()
	### Save predictive model ###
	#joblib.dump(reg, "RF_20.0A_2l3e.pkl", compress=3) 
	
def plot_feature_importances():
	"""A method for plotting feature importances vs. features for Random Forest and Extra Randomized Trees"""
	n_groups =  79
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 1.0
	opacity = 0.8
	x = range(1, 80)
	y =[0.0002093439398878491, 2.4410994580943026e-05, 0.0002493109761396936, 9.98683747337071e-05, 9.433780326304338e-05, 1.223582026719191e-05, 1.965289286862919e-05, 0.06502681614630451, 0.0007136928299262371, 0.0008068166805701582, 0.0008718221055898012, 0.002654611767513143, 0.003974386923685158, 0.019155175475777893, 0.01270670383477641, 0.013395902572617663, 0.01827966229263998, 0.010565632295001135, 0.08678128335241483, 0.015633634819376005, 0.008577717734341978, 0.01175076048729433, 0.01063026110370863, 0.013512025821626856, 0.019730072395866202, 0.006609974104707133, 0.004354576367636046, 0.0, 0.0031537497879282387, 0.002950488747145918, 0.0021749006000659635, 0.0013114829275496672, 0.0013062476677834446, 0.0018506673764636203, 0.0020899277722546353, 0.001711718754510277, 0.001700516836035047, 0.001533174432591087, 0.0012773571436644619, 0.0012747120188143655, 0.0014711940755462553, 0.0016140552701635106, 0.0008383159522521267, 0.0009045187818934073, 0.001052619768052593, 0.0008587357758467248, 0.000985534281522909, 0.0011540803507272955, 0.001087817482267576, 0.0010787128423022183, 0.0007391577355518148, 0.0008390542853929126, 0.0008999022808134895, 0.0008167503206031897, 0.0008215596029099877, 0.000668057117692385, 0.0007126838390759003, 0.0008022765374966769, 0.0009627502006868187, 0.0008114792623283602, 0.0006418709596383171, 0.0008487605079874381, 0.0007253676963116723, 0.0006820688531823465, 0.0012538366735765525, 0.0015334310837728366, 0.0008590543329541559, 0.0011553057521225254, 0.0010338934902666297, 0.0011234363958894584, 0.0010712397199123178, 0.0009488252502020352, 0.0007892273036707022, 0.0007412097226802039, 0.0008800142829735976, 0.0010803665859032085, 0.0010249389860317718, 0.0010777099639448808, 0.6146345526979308]
	rects1 = plt.bar(index, y, bar_width, alpha=opacity, color='b')
	plt.xticks(np.arange(min(x), max(x)+1, 3.0))
	plt.xlabel('Features')
	plt.ylabel('Feature Importances') 
	#plt.ylim(-0.3, 0.3)
	plt.title('Feature Importances vs. Features')
	plt.show()

def main():
	load_set()
	random_forest()
	#plot_feature_importances()

main()
		