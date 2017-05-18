### Load necessary modules ###
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import bisect

#def plot_regression_coefficients():
	#"""A method for plotting regression coefficients vs. features for Linear Regression Model"""
	#n_groups =  78
	#fig, ax = plt.subplots()
	#index = np.arange(n_groups)
	#bar_width = 1.0
	#opacity = 0.8
	#x = range(1, 79)
	#RC_LR = [0.035578166528634894, -0.022350757789094497, -0.005237964288605864, -0.002108197064046359, -0.0021778386844044356, -0.009365119566917938, 0.00566171086431009, 0.0013771479112462725, 0.015084472404053317, -0.026936032867124114, 0.013150800335992913, -0.0012992398729249279, 0.044504022394177015, 4.022996210875651e-05, -3.2066558478321613e-05, 0.0005345775111748816, 0.0035579891919291897, -6.163511505946108e-05, 6.068036860120904e-05, -0.00015246612627392055, -0.003534546926912307, -0.0038451687312895744, -0.008557136576485382, -0.012776056512660026, -0.013097885847051826, 0.030435034300736415, -0.0020130455394923715, 6.591949208711867e-17, 0.05901183983527727, -0.024812628084225713, 0.03425895463952133, -0.04122615384866841, 0.06314919102142505, -0.04938258713498725, 0.043329320871483265, -0.010961226909054458, -0.04396136873584176, -0.07364298729776883, 0.0313402694150998, -0.016585625649253838, 0.1235526607185729, -0.023644731954021565, -0.02351274106541755, -0.05178119585044438, 0.05729372006897175, -0.02958385027212079, 0.0020062172947574236, -0.08341708109470282, 0.043149788635702534, 0.10850469143691605, -0.06913770925218313, 0.05945657587883489, -0.0141419166845278, -0.052152042237053206, 0.023597290540441577, -0.024580854157923465, -0.00018277181852057248, 0.012499475618704407, -0.022232325463430867, 0.01017600973103409, 0.06888076807691156, -0.08663273378479201, 0.13217158851213898, -0.15164719352599068, 0.031280968858538684, -0.022973630269905836, 0.04809604634131227, -0.06758019044538381, 0.06492708574166256, 0.09768826694688092, -0.07581758514922889, 0.03541805981019646, 0.18206722492151775, -0.330311928508516, 0.1534517722568361, -0.09406692770290105, 0.08628057214913389, -0.10578998514743883]
	#y = RC_LR
	#rects1 = plt.bar(index, y, bar_width, alpha=opacity, color='b')
	#plt.xticks(np.arange(min(x), max(x)+1, 3.0))
	#plt.xlabel('Features')
	#plt.ylabel('Regression Coefficients') 
	#plt.ylim(-0.4, 0.4)
	#plt.title('Regression Coefficients vs. Features')
	#plt.show()

#def plot_feature_importances():
	#"""A method for plotting feature importances vs. features for Random Forest and Extra Randomized Trees"""
	#n_groups =  78
	#fig, ax = plt.subplots()
	#index = np.arange(n_groups)
	#bar_width = 1.0
	#opacity = 0.8
	#x = range(1, 79)
	#FI_RF = [0.001449007818023574, 0.0002434888713643932, 0.0001304998609212256, 7.307161867308892e-05, 4.039868631827761e-05, 1.641706681093853e-05, 5.86871267364455e-06, 0.07169055098490099, 0.005169950862488346, 0.0007677873704590405, 0.0021464192757717376, 0.0022311428214420634, 0.07405530875802115, 0.043790024614499455, 0.023261636324786287, 0.018439804619724595, 0.029739879110342364, 0.01441778043160283, 0.10480086402772719, 0.0405104439395107, 0.011039063577764964, 0.010056542361840233, 0.012866052393564373, 0.05080861093842554, 0.019119115107136573, 0.3504632005677362, 0.004093992250936578, 0.0, 0.002370582714156423, 0.0019984070747713317, 0.0014646658102779187, 0.0013609890219832908, 0.001266394043213283, 0.0009551180039993651, 0.0012737790541122022, 0.0008788870584166127, 0.0014462298468412727, 0.0011786611504655128, 0.0009997610675994593, 0.0010345548338249835, 0.0013417861296528824, 0.0024477396678490595, 0.0025466460587044775, 0.005501439654086099, 0.0037345193059332817, 0.0034869835051996693, 0.0029918360085041877, 0.0035616875910494377, 0.001841901732234071, 0.002047356905347874, 0.0018570369589782604, 0.0014293488451544625, 0.0014178996525503942, 0.0058141872980975425, 0.007326094580772929, 0.0026368930807952775, 0.0006439044043149582, 0.00103312459189879, 0.0012812829280880878, 0.0008574171749744404, 0.0015581924416972457, 0.0034190432087307446, 0.0020193541737800816, 0.002180408499510468, 0.005474018408558822, 0.002390279818284525, 0.0010832426652193606, 0.0007011378759541929, 0.0016697682626737665, 0.0014089799677835164, 0.0011754337756603524, 0.0007932248612102896, 0.00207851896345793, 0.0018469071293278798, 0.002590798569198766, 0.0016443491880523237, 0.0026782620182583387, 0.003834041445327223]
	#y = FI_RF
	#rects1 = plt.bar(index, y, bar_width, alpha=opacity, color='b')
	#plt.xticks(np.arange(min(x), max(x)+1, 3.0))
	#plt.xlabel('Features')
	#plt.ylabel('Feature Importances') 
	#plt.ylim(0, 0.38)
	#plt.title('Feature Importances vs. Features')
	#plt.show()
	
### Plotting s2 Order Parameters vs Residue Index ###
def get_s2_vs_resid_plot():
	"""A function for plotting s2 order parameter vs. residue index"""
	#Define x and y variables 
	x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,35]
	#y1 = s2 predicted_RF
	y1 = [0.687312219075777, 0.6943661004795156, 0.7515849678392129, 0.737272282647221, 0.8215767126103718, 0.8406547656200991, 0.7320442379474265, 0.5354438267996616, 0.5802824532642813, 0.44293943184283274, 0.7270267881792141, 0.39061660969620277, 0.8274597611403195, 0.8495168769428717, 0.8256896273484794, 0.743287861905803, 0.8147754538284381, 0.7065477435031956, 0.7901595725588084, 0.6213587603259146, 0.4389705914690694, 0.6575388766479985, 0.8111771828274995, 0.7695437571854762, 0.8212293890370305, 0.8206622434641055, 0.8284779548099925, 0.7540985782143579, 0.780445131383023, 0.7927104332428201, 0.7959440351670449, 0.808788754716105, 0.7473585457168687, 0.7376609052621381, 0.6609796634337203]
	#y2 = s2 predicted_RF/LR
	y2 = [0.7121383341429294, 0.6972554175388738, 0.729768495918158, 0.741784752627513, 0.8053369565863273, 0.8433400327387172, 0.8208781573045285, 0.7078988157395544, 0.5485102735700819, 0.3955808848033043, 0.6238185704492444, 0.31188189048533516, 0.8365010857675856, 0.8242099185967107, 0.8383959253571871, 0.6769117396607147, 0.8434053561093251, 0.6792016923859525, 0.7848181639272651, 0.4647415337391659, 0.6919373596456729, 0.6367853910478839, 0.8265142898129071, 0.779248585188221, 0.8311653110678825, 0.8327856831833371, 0.8326442335870416, 0.7962952716533115, 0.7653249236389664, 0.7634849819316658, 0.7857440721864475, 0.8007229297395267, 0.7734614043658663, 0.7587729818031659, 0.6448267426310815]
	#y3 = s2 measured
	y3 = [0.797367906216, 0.860580382116, 0.85834926324, 0.853475290909, 0.84357737211, 0.85192871712, 0.856036850328, 0.826133435015, 0.761739309437, 0.665135868177, 0.453229570858, 0.201521645996, 0.809433987834, 0.853889974207, 0.865520170334, 0.850794106766, 0.829504396303, 0.817513651735, 0.802298426107, 0.757504276967, 0.758665738462, 0.773312621189, 0.866488403225, 0.874876185351, 0.862149149714, 0.858208068062, 0.866965686516, 0.610199028269, 0.870025692183, 0.874797518299, 0.864267087396, 0.841471668278, 0.842565632659, 0.818741020547, 0.782915125881]
	# Set y-axis boundary
	plt.ylim(0.1, 1.3)
	# Plot lines
	plt.plot(x,y1, marker='o', linestyle='-', color='b') # s2 predicted RF
	plt.plot(x,y2, marker='o', linestyle='-', color='k') # s2 predicted RF/LR
	plt.plot(x, y3, marker='o', linestyle='-', color='r') # s2 measured
	# Title plot and axes
	plt.title('s2 Order Parameters vs. Residue Index')
	plt.xlabel('Residue Index')
	plt.ylabel('s2 Order Parameters')
	# Define and plot legend 
	red_patch = mpatches.Patch(color='red', label='Measured s2 Order Parameters for 2l3e')
	blue_patch = mpatches.Patch(color='blue', label='Predicted s2 Order Parameters for 2l3e (RF)')
	black_patch = mpatches.Patch(color='black', label='Predicted s2 Order Parameters for 2l3e (RF/LR)')
	plt.legend(handles=[red_patch, blue_patch, black_patch])
	# Show plot 
	plt.show()

### Plotting difference in s2 Order Parameters vs Residue Index ###
#def get_s2diff_vs_resid_plot():
	#"""A function for plotting difference in s2 order parameters vs. residue index"""
	# Define x and y variables 
	#x = resid index
	#y1 = (measured - predicted_LR)
	#y2 = (measured - predicted_RF)
	#y3 = (predicted_LR - predicted_RF)
	# Set y-axis boundary 
	#plt.ylim(-0.35, 0.45)
	# Plot lines 
	#plt.axhline(y=0, linewidth=2, color='r')
	#plt.plot(x,y1, marker='o', linestyle='-', color='b')
	#plt.plot(x, y2, marker='o', linestyle='-', color='k')
	#plt.plot(x,y, marker='None', linestyle='-', color='r')
	#blue_line = mlines.Line2D([],[], color='blue', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (LR)')
	#black_line = mlines.Line2D([],[], color='black', marker='o', markersize=8, label='Measured - Predicted s2 Order Parameters (RF)')
	#plt.legend(handles=[blue_line, black_line])
	#plt.title('Difference in s2 Order Parameters vs. Residue Index')
	#plt.xlabel('Residue Index')
	#plt.ylabel('Difference in s2 Order Parameters')
	# Show plot
	#plt.show()

def get_measured_vs_predicted_plot():
 	"""A method for plotting measured vs. predicted s2 order parameters for 2l3e system"""
	#x = measured s2
	x = [0.797367906216, 0.860580382116, 0.85834926324, 0.853475290909, 0.84357737211, 0.85192871712, 0.856036850328, 0.826133435015, 0.761739309437, 0.665135868177, 0.453229570858, 0.201521645996, 0.809433987834, 0.853889974207, 0.865520170334, 0.850794106766, 0.829504396303, 0.817513651735, 0.802298426107, 0.757504276967, 0.758665738462, 0.773312621189, 0.866488403225, 0.874876185351, 0.862149149714, 0.858208068062, 0.866965686516, 0.610199028269, 0.870025692183, 0.874797518299, 0.864267087396, 0.841471668278, 0.842565632659, 0.818741020547, 0.782915125881, 0.835262157743, 0.849284670213, 0.859660505959, 0.860814379415, 0.847453251849, 0.831138948291, 0.815544812019, 0.799283071469, 0.811626239916, 0.81472899248, 0.698090605191, 0.57109343066, 0.470150178151, 0.362028396382, 0.818951136094, 0.826512552458, 0.82805558862, 0.806618291692, 0.791630762989, 0.57520474948, 0.710715821944, 0.82653742244, 0.83335132482, 0.827892511079, 0.806840390225, 0.814897676407, 0.806696144196, 0.807980407878, 0.814215193123, 0.812090614628, 0.720134378141, 0.572074815485, 0.493056803336, 0.200326011493, 0.843458938273, 0.856641434625, 0.811134529589, 0.844944905513, 0.761653846319, 0.49244675259, 0.753542918394, 0.846242838066, 0.847078897902, 0.82179146662, 0.824002668095, 0.826231136339, 0.832038711102, 0.828714542151, 0.861112734155, 0.850526715075, 0.812798696368, 0.84822100404, 0.822728202034, 0.874904394505, 0.867206635447, 0.854837179833, 0.865127594001, 0.862650338333, 0.8523181086, 0.846256226026, 0.80849970126, 0.810912551567, 0.783033000952, 0.850114682817, 0.842083972225, 0.790685363269, 0.851955697785, 0.869741947045, 0.849369708369, 0.833262831842, 0.839436763634, 0.841538999829, 0.837028010192, 0.838615856332, 0.551313568752, 0.828643820433, 0.780694820364, 0.459574067645]
 	#y = measured s2
 	y = [0.797367906216, 0.860580382116, 0.85834926324, 0.853475290909, 0.84357737211, 0.85192871712, 0.856036850328, 0.826133435015, 0.761739309437, 0.665135868177, 0.453229570858, 0.201521645996, 0.809433987834, 0.853889974207, 0.865520170334, 0.850794106766, 0.829504396303, 0.817513651735, 0.802298426107, 0.757504276967, 0.758665738462, 0.773312621189, 0.866488403225, 0.874876185351, 0.862149149714, 0.858208068062, 0.866965686516, 0.610199028269, 0.870025692183, 0.874797518299, 0.864267087396, 0.841471668278, 0.842565632659, 0.818741020547, 0.782915125881, 0.835262157743, 0.849284670213, 0.859660505959, 0.860814379415, 0.847453251849, 0.831138948291, 0.815544812019, 0.799283071469, 0.811626239916, 0.81472899248, 0.698090605191, 0.57109343066, 0.470150178151, 0.362028396382, 0.818951136094, 0.826512552458, 0.82805558862, 0.806618291692, 0.791630762989, 0.57520474948, 0.710715821944, 0.82653742244, 0.83335132482, 0.827892511079, 0.806840390225, 0.814897676407, 0.806696144196, 0.807980407878, 0.814215193123, 0.812090614628, 0.720134378141, 0.572074815485, 0.493056803336, 0.200326011493, 0.843458938273, 0.856641434625, 0.811134529589, 0.844944905513, 0.761653846319, 0.49244675259, 0.753542918394, 0.846242838066, 0.847078897902, 0.82179146662, 0.824002668095, 0.826231136339, 0.832038711102, 0.828714542151, 0.861112734155, 0.850526715075, 0.812798696368, 0.84822100404, 0.822728202034, 0.874904394505, 0.867206635447, 0.854837179833, 0.865127594001, 0.862650338333, 0.8523181086, 0.846256226026, 0.80849970126, 0.810912551567, 0.783033000952, 0.850114682817, 0.842083972225, 0.790685363269, 0.851955697785, 0.869741947045, 0.849369708369, 0.833262831842, 0.839436763634, 0.841538999829, 0.837028010192, 0.838615856332, 0.551313568752, 0.828643820433, 0.780694820364, 0.459574067645]
 	#y1 = predicted LR/RF s2
 	y1 = [0.7121383341429294, 0.6972554175388738, 0.729768495918158, 0.741784752627513, 0.8053369565863273, 0.8433400327387172, 0.8208781573045285, 0.7078988157395544, 0.5485102735700819, 0.3955808848033043, 0.6238185704492444, 0.31188189048533516, 0.8365010857675856, 0.8242099185967107, 0.8383959253571871, 0.6769117396607147, 0.8434053561093251, 0.6792016923859525, 0.7848181639272651, 0.4647415337391659, 0.6919373596456729, 0.6367853910478839, 0.8265142898129071, 0.779248585188221, 0.8311653110678825, 0.8327856831833371, 0.8326442335870416, 0.7962952716533115, 0.7653249236389664, 0.7634849819316658, 0.7857440721864475, 0.8007229297395267, 0.7734614043658663, 0.7587729818031659, 0.6448267426310815, 0.8220863941730109, 0.7243308918048683, 0.7365567574091458, 0.7637531125126232, 0.7851105246604216, 0.7097286071978934, 0.7231804548703544, 0.7840173053525896, 0.8247547423638073, 0.8083455427543921, 0.4755653252295166, 0.3458302346641018, 0.6284080710254843, 0.267477242431892, 0.8252635456628569, 0.6915919689460385, 0.7357467955915769, 0.6949232925025192, 0.7580869295413741, 0.3591696807122638, 0.6917483138797224, 0.7179874454362897, 0.8202256061887782, 0.7384972692287154, 0.6402013404946135, 0.7150999663852422, 0.739995790968499, 0.6713933134324157, 0.8359102574443973, 0.8111281422546777, 0.5422367381533488, 0.32408849304044524, 0.567719509833182, 0.2720006378661221, 0.8270525776586908, 0.8126806240775404, 0.6604068085285464, 0.6820739839234166, 0.7692866203755269, 0.3500875090846723, 0.7126620726425701, 0.7564880427558277, 0.8029974258119765, 0.7376107629541687, 0.6406946729919433, 0.7058631670246908, 0.6860697146291171, 0.7260185640834572, 0.8310346403692255, 0.8454330608807114, 0.5121259283899215, 0.8070179001199473, 0.8293824697051584, 0.8386298977508648, 0.7959629812767722, 0.7171382339856016, 0.7351490790876115, 0.7599440068060995, 0.7827227947137679, 0.7527899665661502, 0.6868491996856164, 0.6815735148787619, 0.6889812078622551, 0.8352526428261418, 0.8449955479997149, 0.6270084048645477, 0.8083473799886458, 0.8266349140949075, 0.7879259234627586, 0.7570415000720971, 0.735206231763514, 0.7948526200390781, 0.8326384587490969, 0.8076261715049442, 0.3957385080446724, 0.7553340265607722, 0.7680902174696148, 0.35327367061531256]
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
	
def main():
	#plot_regression_coefficients()
	#plot_feature_importances()
	#get_s2_vs_resid_plot()
	get_measured_vs_predicted_plot()

main()
 