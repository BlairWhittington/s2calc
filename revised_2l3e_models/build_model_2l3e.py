import build_linear_model_2l3e
from build_linear_model_2l3e import *
import build_random_forest_model_2l3e
from build_random_forest_model_2l3e import *

def main():
	### Use for Linear Model ###
	linear = build_linear_model()
	linear.load_set()
	linear.linear_regression()
	#linear.compute_rmse()
	#linear.compute_explained_variance()
	linear.get_coefficients()
	linear.get_predicted_s2()
	linear.get_s2_diff()
	#linear.get_kendall_tau()
	#linear.make_table()

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
	#linear.plot_features()
	#random.plot_features()

	### Use for (C1', H1') plots ###
	#linear.plot_s2_resid_C1()
	#linear.plot_s2_diff_resid_C1()
	#random.plot_s2_resid_C1()
	#random.plot_s2_diff_resid_C1()

	### Use for (C2, H2) plots ###
	#linear.plot_s2_resid_C2()
	#linear.plot_s2_diff_resid_C2()
	#random.plot_s2_resid_C2()
	#random.plot_s2_diff_resid_C2()

	### Use for (C5, H5) plots ###
	#linear.plot_s2_resid_C5()
	#linear.plot_s2_diff_resid_C5()
	#random.plot_s2_resid_C5()
	#random.plot_s2_diff_resid_C5()

	### Use for (C6, H6) plots ###
	#linear.plot_s2_resid_C6()
	#linear.plot_s2_diff_resid_C6()
	#random.plot_s2_resid_C6()
	#random.plot_s2_diff_resid_C6()

	### Use for (C8, H8) plots ###
	#linear.plot_s2_resid_C8()
	#linear.plot_s2_diff_resid_C8()
	#random.plot_s2_resid_C8()
	#random.plot_s2_diff_resid_C8()

	### Use for (N1, H1) plots ###
	#linear.plot_s2_resid_N1()
	#linear.plot_s2_diff_resid_N1()
	#random.plot_s2_resid_N1()
	#random.plot_s2_diff_resid_N1()

	### Use for (N3, H3) plots ###
	#linear.plot_s2_resid_N3()
	#linear.plot_s2_diff_resid_N3()
	#random.plot_s2_resid_N3()
	#random.plot_s2_diff_resid_N3()

	### Use for all bond vector plots ###
 	#linear.plot_mesaured_predicted_s2()
 	#random.plot_measured_predicted_s2()

if __name__ == "__main__":
	main()
