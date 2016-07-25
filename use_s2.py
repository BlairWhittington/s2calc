import s2_calculator
from s2_calculator import s2_calculator
import MDAnalysis 

PSF = "1R7Z.nomin.psf"
DCD = "pool.dcd"
u = initialize_universe(PSF, DCD)

t = ("C1'", "H1'")
nframes = len(u.trajectory)
nresid = len(u.atoms.residues)
s2 = s2_calculator(u, t)
s2.get_all_s2()
print s2.s2_list
s2.get_scatterplot()
s2.get_table()
