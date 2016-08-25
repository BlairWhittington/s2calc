### Calculate s2 order parameters ###
# Check to see if you are in correct directory (/Users/blairwhittington/Downloads/GarrettAndZach/*/struct)
# Load python interpreter
#ipython	 --matplotlib
import numpy as numpy
# Load MDAnalysis
import MDAnalysis 
from MDAnalysis.tests.datafiles import PSF, DCD
u = MDAnalysis.Universe("1R7Z.nomin.psf", "pool.dcd")
nframes = len(u.trajectory)

# Normalize vectorCH
def norm_vec(v):
	"""get vector norm"""
	return (numpy.sqrt(numpy.sum(v * v)))
	
def normalize_vec(v):
	"""normalize vector"""
	vecCH = (v / norm_vec(v))
	return vecCH

# Define a function to calculate s2 values 
def s2_calculator(i=1):
    # Set initial vector quantities equal to zero 
    x2 = 0
    y2 = 0
    z2 = 0
    xy = 0
    xz = 0
    yz = 0
    # Make carbon atom selection
    i = 1
    sel1 = u.select_atoms("resid %s and name C1'" % (i))
    # Check to see if a carbon atom has been selected
    if sel1.n_atoms != 1:
        print "error must select 1 carbon atom" 
    # Make hydrogen atom selection  
    sel2 = u.select_atoms("resid %s and name H1'" % (i))
    # Check to see if a hydrogen atom has been selected
    if sel2.n_atoms != 1:
        print "error must select 1 hydrogen atom"
    # Loop over trajectory
    for ts in u.trajectory:
        sel1 = u.select_atoms("resid %s and name C1'" % (i))
        if sel1.n_atoms != 1:
            print "error must select 1 carbon atom" 
        sel2 = u.select_atoms("resid %s and name H1'" % (i))
        if sel2.n_atoms != 1:
            print "error must select 1 carbon atom" 
        # Define residue number for selected atoms
        resnum = u.select_atoms("resid %s and name C1'" % (i) and "resid %s and name H1'" % (i)).residues.resids
        # Define vectorCH
        vecCH = sel1.center_of_mass() - sel2.center_of_mass()
        import numpy
        vecCH = normalize_vec(vecCH) 
        # Get vector components
        xcomp = vecCH[0]
        ycomp = vecCH[1]
        zcomp = vecCH[2]    
        # Iterate through vector components
        x2 += xcomp * xcomp
        y2 += ycomp * ycomp
        z2 += zcomp * zcomp
        xy += xcomp * ycomp
        xz += xcomp * zcomp
        yz += ycomp * zcomp
    # Calculate vector averages
    x2 = x2 / nframes
    y2 = y2 / nframes
    z2 = z2 / nframes

    xy = xy / nframes
    xz = xz / nframes
    yz = yz / nframes
    
    # Calculate s2 
    s2 = (1.5 * ((x2 ** 2) + (y2 ** 2) + (z2 ** 2))) + (3 * ((xy ** 2) + (xz ** 2) + (yz ** 2))) - 0.5
    return s2
