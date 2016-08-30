README file for predicting the dynamics of static RNA structures. 

----------------------
  MDAnalysis Package
----------------------
A python package used to evaluate MD simulations
pipe install --upgrade MDAnalysis
conda config --add channels MDAnalysis
conda install mdanalysis 


/// Compute s2 order parameters for all bond vector selections ///
cd /Users/blairwhittington/Documents/Projects/s2calc/scripts
vi s2_calculator.py

/// Compute contact sum between heavy atoms for all bond vector selections ///
cd Users/blairwhittington/Documents/Projects/s2calc/scripts/contact_sum.py
vi contact_sum.py

/// Edit use_s2.py ///
Set r_cut and r_eff variables and bond vector selections in use_s2.py
cd /Users/blairwhittington/Documents/Projects/s2calc/scripts
vi use_s2.py

/// Run scripts /// 
cd /Users/blairwhittington/Documents/Projects/s2calc
.scripts/get_s2_Ci.sh
vi *.txt 

