import s2_calculator
from s2_calculator import *
import contact_sum
from contact_sum import *
import MDAnalysis 
from itertools import repeat
from optparse import OptionParser
import warnings, sys

PSF="../test/1R7Z.nomin.psf"
DCD0="../test/pool.dcd"

u = initialize_universe(PSF, DCD0)

#t = [("C1'", "H1'"), ("C2", "H2"), ("C5", "H5"), ("C6", "H6"), ("C8", "H8"), ("N1", "H1"), ("N3", "H3")]
t = [("C1'", "H1'")]
nframes = len(u.trajectory)
nresid = len(u.atoms.residues)    

### Use for s2_calculator ###
s2 = s2_calculator(u, t)
s2.get_all_s2()
#s2.get_scatterplot()

### Use for contact_sum ###
n=50
r_cut=20.0
r_eff=5.0
sum = contact_sum(u, t, n, r_cut, r_eff)
sum.get_all_distances()
sum.distance_list
