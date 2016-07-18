import s2_calculator
from s2_calculator import s2_calculator
import MDAnalysis 

u = MDAnalysis.Universe("1R7Z.nomin.psf", "pool.dcd")
nframes = len(u.trajectory)
s2 = s2_calculator(u)
s2.get_s2(1)
s2.get_all_s2()

