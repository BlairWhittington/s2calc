### Class for calculating s2 order parameters ###
# Check to see if you are in correct directory (/Users/blairwhittington/Downloads/GarrettAndZach/*/struct)
# Load python interpreter
ipython	--matplotlib
import numpy as numpy
# Load MDAnalysis
import MDAnalysis 
from MDAnalysis.tests.datafiles import PSF, DCD
nframes = len(u.trajectory)

# Normalize vectorCH	
def norm_vec(v):
	"""get vector norm"""
	return (numpy.sqrt(numpy.sum(v * v)))
	
def normalize_vec(v):
	"""normalize vector"""
	vecCH = (v / norm_vec(v))
	return vecCH

# Assign the variable u to the Universe
u = MDAnalysis.Universe("1R7Z.nomin.psf", "pool.dcd")

# Create a s2_calculator class that takes information from universe 
class s2_calculator(u):
	# A method for Initializing s2 order parameter, and residue number
	def __init__(self, s2):
		self.s2 = s2
		self.s2_list = []
		
	def get_s2(self, i=1):
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
			# Define vector CH
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

	# A method for computing s2 order parameters for all residues
	def get_alls2(self):
		for i in range(len(u.atoms.residues.resids)):
			get_s2(i)
		self.s2_list.append(get_s2(i))
		
		# A list of resids
		resid_list = u.atoms.residues.resids

	# A method for plotting s2 vs. residue number
	# Plot of s2 order parameters versus residue number 
	from matplotlib import pyplot
	from numpy import arange
	import bisect
	# Create a scatterplot of s2 order parameters vs. residue number
	def get_scatterplot(x,y):
		# Assign the variable x to list of resids
		x = resid_list
		# Assign the variable y to list of s2 order parameters
		y = s2_list
		# Plot the assigned variables as dots
		pyplot.plot(x,y,'.')
		# Label the x-axis Residue Number
		pyplot.xlabel('Residue Number')
		# Label the y-axis s2 Order Parameter
		pyplot.ylabel('s2 Order Parameter')
		# Show the plot 
		pyplot.show()

		

	