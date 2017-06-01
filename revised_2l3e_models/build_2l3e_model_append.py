import build_random_forest_model_2l3e
from build_random_forest_model_2l3e import *

def main():
	### Use for Random Forest Model ###
	random = build_random_forest_model()
	random.load_set()
	random.random_forest()
	#random.compute_rmse()
	#random.compute_explained_variance()
	random.get_feature_importances()
	random.get_predicted_s2()
	random.get_s2_diff()
	#random.get_kendall_tau()
	#random.make_table()

	### Use to plot Features ###
	#random.plot_features()

	### Use for (C1', H1') plots ###
	#random.plot_s2_resid_C1()
	random.plot_s2_diff_resid_C1()

	### Use for (C2, H2) plots ###
	#random.plot_s2_resid_C2()
	#random.plot_s2_diff_resid_C2()

	### Use for (C5, H5) plots ###
	#random.plot_s2_resid_C5()
	#random.plot_s2_diff_resid_C5()

	### Use for (C6, H6) plots ###
	#random.plot_s2_resid_C6()
	#random.plot_s2_diff_resid_C6()

	### Use for (C8, H8) plots ###
	#random.plot_s2_resid_C8()
	#random.plot_s2_diff_resid_C8()

	### Use for (N1, H1) plots ###
	#random.plot_s2_resid_N1()
	#random.plot_s2_diff_resid_N1()

	### Use for (N3, H3) plots ###
	#random.plot_s2_resid_N3()
	#random.plot_s2_diff_resid_N3()

	### Use for all bond vector plots ###
 	#random.plot_measured_predicted_s2()

if __name__ == "__main__":
	main()
