import build_linear_model
from build_linear_model import *
import build_random_forest_model
from build_random_forest_model import *
import build_extra_trees_model
from build_extra_trees_model import *

def main():
	
	### Use for Linear Model ###
	linear = build_linear_model()
	linear.load_set()
	linear.linear_regression()
	linear.compute_rmse()
	linear.compute_explained_variance()
	linear.get_coefficients()
	linear.get_predicted_s2()
	#linear.make_table()
	
	### Use for Random Forest Model ###
	random = build_random_forest_model()
	random.load_set()
	random.random_forest()
	random.compute_rmse()
	random.compute_explained_variance()
	random.get_feature_importances()
	random.get_predicted_s2()
	#random.make_table()
	
	### Use for Extra Randomized Trees Model ###
	extra = build_extra_trees_model()
	extra.load_set()
	extra.extra_trees()
	extra.compute_rmse()
	extra.compute_explained_variance()
	extra.get_feature_importances()
	extra.get_predicted_s2()
	#extra.make_table()
		
	### Use to Plot ###	
	linear.plot_features()
	random.plot_features()
	extra.plot_features()
    
if __name__ == "__main__":
    main()