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
DCD0="/Users/atfrank/GitSoftware/s2calc/data/2l3e/pool.dcd"
universe = MDAnalysis.Universe(PSF, DCD0)
new = Features(universe)
new.unique_HBtable()
new.ts = 0
new.loadCoordinateDependentDataStructures()
new.residueLevelFeaturesCompute()
for key in range(1,35):
	print key, new.system['%i'%key]

