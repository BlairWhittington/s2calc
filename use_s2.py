import s2_calculator
from s2_calculator import *
import contact_sum
from contact_sum import *
import MDAnalysis 

PSF = "1R7Z.nomin.psf"
DCD0 = "pool.dcd"

u = initialize_universe(PSF, DCD0)

t = ("C1'", "H1'")

nframes = len(u.trajectory)
nresid = len(u.atoms.residues)

### Use for s2_calculator ###
s2 = s2_calculator(u, t)
s2.get_all_s2()
print s2.s2_list
#s2.get_scatterplot()
	
### Use for contact_sum ###
r_cut=5.0
r_eff=5.0
sum = contact_sum(u, t, r_cut, r_eff)
sum.get_all_contact_sums()
#print sum.Ci_list
#sum.get_contact_sum_one()	
#sum.get_contact_sum_two()
	
			
### For profiling contact_sum functions ###	
#import cProfile
#cProfile.run('sum.get_contact_sum_one()')
#cProfile.run('sum.get_contact_sum_two()')


### Table of resids, s2 order parameters, Ci values, and atom selections ###
s2.resid_list = [ '%.2i' % i for i in s2.resid_list]
s2.s2_list = [ '%.8f' % i for i in s2.s2_list]
sum.Ci_list = ['%.8f' % i for i in sum.Ci_list]
for i, v, f in zip(s2.resid_list, s2.s2_list, sum.Ci_list):
	print i, v, f, t[0], t[1]	
