### Alternative method for computing contact sum
import MDAnalysis.lib.distances
from MDAnalysis.lib.distances import distance_array
import numpy as numpy

u = MDAnalysis.Universe("1R7Z.nomin.psf", "pool.dcd")
nframes = len(u.trajectory)
t = ("C1'", "H1'")

class contact_sum:
	def __init__(self, u, t):
		"""Initializing the class variables that will be used later"""
		self.u = u
		self.t = t
		self.res_list = []
		self.Ci_list = []
	
	# Compute the distances between only the heavy atoms in the trajectory	
	def get_heavy_atom_distances(self):
		self.heavy_atoms = self.u.select_atoms("not type 102 103 104 109 111 112") 
		dismat = distance_array(self.heavy_atoms.coordinates(), self.heavy_atoms.coordinates())
	
	# Get carbon atom indices
	def get_carbon_atom_index(self):
		""" A method for getting the carbon atom indices"""
		# Find carbon atom selection index 
		carbon_primes = self.u.select_atoms("name %s" % (self.t[0]))
		n_atoms = len(carbon_primes)
		for i in range(n_atoms):
			index = carbon_primes.atoms[0].index
			resid = carbon_primes.atoms[0].resid
			# Returns the index of carbon atom selection 
			dist = dismat[:, index]
			
	def get_contact_sum(self, r_eff=5.0, r_cut=5.0):
		"""A method for getting sum of distances between carbon selection and all other
		heavy atoms within 5.0A"""
		# Update distance matrix
		for i in dist:
			if i != 0.0 and i < r_cut:
				dist_list.append(i)
				
		for i in dist_list:
			Ci = numpy.exp(-1 * (i / r_eff))
			Ci_list = []
			Ci_list.append(Ci)
			Ci = numpy.sum(Ci_list)
			return Ci	
			

	
	
	
	
