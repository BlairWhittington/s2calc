import s2_calculator
from s2_calculator import s2_calculator
import MDAnalysis 

u = MDAnalysis.Universe("1R7Z.nomin.psf", "pool.dcd")
t = ("C8", "H8")
nframes = len(u.trajectory)
s2 = s2_calculator(u, t)
s2.get_all_s2()
s2.get_scatterplot()
