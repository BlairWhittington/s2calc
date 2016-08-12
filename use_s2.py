import s2_calculator
from s2_calculator import *
import contact_sum
from contact_sum import *
#import alternative_contact_sum
#from alternative_contact_sum import *
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
	
# Use for contact_sum function
r_cut=5.0
r_eff=5.0
sum = contact_sum(u, t, r_cut, r_eff)
sum.get_contact_sum()	


#sum.time_list = [ '%. f' % for i in sum.time_list]
#for i, v in zip(sum.cutoff_list, sum.time_list):
	#print i, v

# Use for alternative_contact_sum function
#get_contact_sum(u, r_cut=5.0, r_eff=5.0)




