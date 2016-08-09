import s2_calculator
from s2_calculator import *
import contact_sum
from contact_sum import *
#import contact_sum_method_2
#from contact_sum_method_2 import *
import MDAnalysis 

PSF = "1R7Z.nomin.psf"
DCD0 = "pool.dcd"

u = initialize_universe(PSF, DCD0)

t = ("C1'", "H1'")

nframes = len(u.trajectory)
nresid = len(u.atoms.residues)


s2 = s2_calculator(u, t)
s2.get_all_s2()
print s2.s2_list
s2.get_scatterplot()


s2.resid_list = [ '%.2i' % i for i in s2.resid_list]
s2.s2_list = [ '%.8f' % i for i in s2.s2_list]
for i, v in zip(s2.resid_list, s2.s2_list):
	print i , v , t[0], t[1]
	

sum = contact_sum(u, t)
sum.get_contact_sum()

#sum = contact_sum(u, t)
#sum.get_carbon_atom_index()
#sum.get_contact_sum()
