### Calculate s2 order parameters ###
# Check to see if you are in correct directory (/Users/blairwhittington/Downloads/GarrettAndZach/*/struct)
# Load python interpreter
#ipython	 --matplotlib
import numpy as numpy
# Load MDAnalysis
import MDAnalysis 
from MDAnalysis.tests.datafiles import PSF, DCD

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

# Load universe and define frames 
u = MDAnalysis.Universe("1R7Z.nomin.psf", "pool.dcd")
nframes = len(u.trajectory)
resi_list = []
s2_list = []
# TODO: determine number residue in u, the residue numbers
#       then loop over those
	s2_list.append(s2_calculator(i))
 	resi_list.append(i)
	

# Plot of s2 order parameters versus residue number 
from matplotlib import pyplot
from numpy import arange
import bisect
# Create a scatterplot of s2 order parameters vs. residue number
def stick_plot(x,y):
	# Assign the variable x to residue number
	x = resnum
	# Assign the variable y to s2 order parameter
	y = s2
	# Plot the assigned variables as dots
	pyplot.plot(x,y,'.')
	# Label the x-axis Residue Number
	pyplot.xlabel('Residue Number')
	# Label the y-axis s2 Order Parameter
	pyplot.ylabel('s2 Order Parameter')
	# Show the plot 
	pyplot.show()

   #def get_table(self):
        #"""A method for listing s2 order parameters by resids and bond vectors in a table"""
        #from astropy.table import Table, Column
        #import numpy as np
        #from itertools import repeat
        # Define a column resid that stores information from self.resid_list
        #resid = self.resid_list
        # Define a column s2 that stores information from self.s2_list
        #s2 = self.s2_list
        # Define carbon as a list containing carbon atom selection repeated once for each residue
        #self.carbon_list.extend(repeat(self.t[0], self.nresid))
        #carbon = self.carbon_list
        # Define hydrogen as a list containing hydrogen atom selection repeated once for each residue 
        #self.hydrogen_list.extend(repeat(self.t[1], self.nresid))
        #hydrogen = self.hydrogen_list
        # Create a table containing resid, s2 order parameter, carbon and hydrogen atoms selection information 
        #s = Table([resid, s2, carbon, hydrogen], names = ("resid", "s2", "carbon_selection", "hydrogen_selection"))
        #s
