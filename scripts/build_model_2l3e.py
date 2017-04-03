# Load necessary modules 
import numpy as np
import math
from sklearn import linear_model
from sklearn.externals import joblib

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
### Compute RMSE ###
print(math.sqrt(np.mean((reg.predict(x_test)-y_test)**2)))
### Compute explained variance ###
print(reg.score(x_test, y_test))
### Compute feature importances ###
print(reg.coef_)
### Predict s2 order parameters ###
s2 = reg.predict(x_test) #predicted s2 values for 2l3e only  
s2.tolist()
print s2
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
y1 = predicted_s2
y2 = computed_s2
plt.title('s2 Order Parameters vs. Residue Index')
plt.plot(x,y1, marker='o', linestyle='-', color='r')
plt.plot(x,y2, marker='o', linestyle='-', color='b')
plt.xlabel('Residue Index')
plt.ylabel('s2 Order Parameters')
red_patch = mpatches.Patch(color='red', label='Predicted s2 Order Parameters for 2l3e')
blue_patch = mpatches.Patch(color='blue', label='Computed s2 Order Parameters for 2l3e')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

### Plotting change in s2 Order Parameters vs Residue Index ###
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
x = resid_index
y = np.subtract(y2, y1)
y = y.tolist()
plt.title('Difference in s2 Order Parameters vs. Residue Index')
plt.plot(x,y, marker='o', linestyle='-', color='b')
plt.plot(x,y3, marker='None', linestyle='-', color='r')
blue_line = mlines.Line2D([],[], color='blue', marker='o', markersize=8, label='Actual - Predicted s2 Order Parameters')
plt.legend(handles=[blue_line])
plt.xlabel('Residue Index')
plt.ylabel('Difference in s2 Order Parameters')
plt.show()

### Plotting Actual s2 order parameters vs. Predicted s2 order parameters ###
# Actual 2l3e s2 order parameters
x = [0.797367906216, 0.860580382116, 0.85834926324, 0.853475290909, 0.84357737211, 0.85192871712, 0.856036850328, 0.826133435015, 0.761739309437, 0.665135868177, 0.453229570858, 0.201521645996, 0.809433987834, 0.853889974207, 0.865520170334, 0.850794106766, 0.829504396303, 0.817513651735, 0.802298426107, 0.757504276967, 0.758665738462, 0.773312621189, 0.866488403225, 0.874876185351, 0.862149149714, 0.858208068062, 0.866965686516, 0.610199028269, 0.870025692183, 0.874797518299, 0.864267087396, 0.841471668278, 0.842565632659, 0.818741020547, 0.782915125881, 0.835262157743, 0.849284670213, 0.859660505959, 0.860814379415, 0.847453251849, 0.831138948291, 0.815544812019, 0.799283071469, 0.811626239916, 0.81472899248, 0.698090605191, 0.57109343066, 0.470150178151, 0.362028396382, 0.818951136094, 0.826512552458, 0.82805558862, 0.806618291692, 0.791630762989, 0.57520474948, 0.710715821944, 0.82653742244, 0.83335132482, 0.827892511079, 0.806840390225, 0.814897676407, 0.806696144196, 0.807980407878, 0.814215193123, 0.812090614628, 0.720134378141, 0.572074815485, 0.493056803336, 0.200326011493, 0.843458938273, 0.856641434625, 0.811134529589, 0.844944905513, 0.761653846319, 0.49244675259, 0.753542918394, 0.846242838066, 0.847078897902, 0.82179146662, 0.824002668095, 0.826231136339, 0.832038711102, 0.828714542151, 0.861112734155, 0.850526715075, 0.812798696368, 0.84822100404, 0.822728202034, 0.874904394505, 0.867206635447, 0.854837179833, 0.865127594001, 0.862650338333, 0.8523181086, 0.846256226026, 0.80849970126, 0.810912551567, 0.783033000952, 0.850114682817, 0.842083972225, 0.790685363269, 0.851955697785, 0.869741947045, 0.849369708369, 0.833262831842, 0.839436763634, 0.841538999829, 0.837028010192, 0.838615856332, 0.551313568752, 0.828643820433, 0.780694820364, 0.459574067645]
# Predicted 2l3e s2 order parameters
y = [0.6384418447989992, 0.8334062959431285, 0.8378405896281069, 0.7191664900095447, 0.7365458088737709, 0.7888370941796452, 0.7411246293713825, 0.7001611273366437, 0.5760819907478065, 0.4414679112322862, 0.6439058726785942, 0.33313186611131523, 0.7868219267121075, 0.7640205236415312, 0.8714147236804146, 0.6801799803161701, 0.842158525403568, 0.6595779458962869, 0.7272586109951267, 0.4546374387498549, 0.6147569551434617, 0.5533953239333165, 0.8809345158553865, 0.7625756295645274, 0.8623443616119395, 0.8344813076011455, 0.9287273670112455, 0.8531697828337437, 0.7854116456526989, 0.8782824093406976, 0.8806280921395488, 0.8702216408227617, 0.8938667203413952, 0.7801421362731886, 0.7419864081349459, 0.89285718430354, 0.7501011047974606, 0.8715270449935779, 0.8419271783632059, 0.8519885206829716, 0.8066289387212897, 0.6752119485258457, 0.7249117714714697, 0.7237392353569219, 0.7682433945617306, 0.5375660184790483, 0.34827656913424115, 0.6254519911275558, 0.24393267566957438, 0.7904397072982186, 0.661591876749221, 0.7079403705909868, 0.6979317771203626, 0.6783460374948758, 0.41132016126683074, 0.6152840972192706, 0.7013152616684548, 0.8048554844859341, 0.7607425176740994, 0.7129520560910103, 0.8142049022757292, 0.7252434204431359, 0.6642270484841075, 0.7558973159830253, 0.7218574924606445, 0.5525581063506813, 0.31259145396146143, 0.5824018701188407, 0.32028535372253925, 0.7707411692664368, 0.7543917144488856, 0.6573314359242965, 0.6646895439266782, 0.6976883856474394, 0.40525629765170096, 0.6765352051348363, 0.7155214781022872, 0.7609378224070558, 0.7572736962981208, 0.7748401383155892, 0.6607726165870524, 0.8393281087487521, 0.7041274472228813, 0.7896456627526589, 0.7942919176504046, 0.5108615482028174, 0.7764855153974757, 0.85357163938772, 0.907412238982858, 0.846130925395945, 0.6780701735741743, 0.8696475753356931, 0.8328071798860657, 0.8375301721346229, 0.8838060564485406, 0.7308973260287572, 0.7673456518524733, 0.6901023169762694, 0.8411134982655522, 0.8267852765964008, 0.5587215414901989, 0.7906790360141718, 0.8673225653677259, 0.823041648023976, 0.8664506660894495, 0.7385888641802032, 0.7374650946855497, 0.7615120695135051, 0.7608785642967899, 0.35125137029624015, 0.7434542296251079, 0.7073653068667858, 0.360162630269054]
# Title Plot
plt.title('Actual vs. Predicted s2 Order Parameters for 2l3e')
# (C1', H1')
plt.plot(x[0:35],y[0:35], marker='o', linestyle='None', color='r', label="(C1', H1')")
# (C2, H2)
plt.plot(x[35:40],y[35:40], marker='o', linestyle='None', color='k', label="(C2, H2)")
# (C5, H5)
plt.plot(x[40:60],y[40:60], marker='o', linestyle='None', color='g', label="(C5, H5)")
# (C6, H6)
plt.plot(x[60:80],y[60:80], marker='o', linestyle='None', color='b', label="(C6, H6)")
# (C8, H8)
plt.plot(x[80:95],y[80:95], marker='o', linestyle='None', color='m', label="(C8, H8)")
# (N1, H1)
plt.plot(x[95:105],y[95:105], marker='o', linestyle='None', color='y', label="(N1, H1)")
# (N3, H3)
plt.plot(x[105:113],y[105:113], marker='o', linestyle='None', color='c', label="(N3, H3)")
# Create Legend
plt.legend(bbox_to_anchor=(0.05, 0.7), loc=6, borderaxespad=0.)
# Label x-axis
plt.xlabel('Actual 2l3e s2 Order Parameters')
# Label y-axis
plt.ylabel('Predicted 2l3e s2 Order Parameters')
# Show Plot
plt.show()