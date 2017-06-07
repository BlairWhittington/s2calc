import build_bond_vector_predictor
from build_bond_vector_predictor import *

def main():
	predictor = build_bond_vector_predictor() 
	predictor.load_set()
	predictor.linear_regression()
	predictor.random_forest()
	predictor.extra_trees()
	predictor.compute_rmse()
	predictor.compute_explained_variance()
	predictor.get_features()
	predictor.get_predicted_s2()
	predictor.get_s2_diff()
	predictor.get_kendall_tau()
	predictor.make_table()

	### Use for LR Predictor ###
	#predictor.plot_features_LR()

	### Use for RF Predictor ###
	#predictor.plot_features_RF()

	### Use for ET Predictor ###
	#predictor.plot_features_ET()
	
if __name__ == "__main__":
    main()