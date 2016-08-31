import s2_calculator
from s2_calculator import *
import contact_sum
from contact_sum import *
import MDAnalysis 
from itertools import repeat
from optparse import OptionParser
import warnings, sys
import Features
from Features import Features

#initialize variables from the commandline

PSF="/Users/atfrank/GitSoftware/s2calc/data/2l3e/reference.psf"
DCD0="/Users/atfrank/GitSoftware/s2calc/data/2l3e/test.dcd"
universe = initialize_universe(PSF, DCD0)
new = Features(universe)
for ts in universe.trajectory:
		new.ts = ts.frame-1
		new.loadCoordinateDependentDataStructures()
		new.residueLevelFeaturesCompute()


647-3983