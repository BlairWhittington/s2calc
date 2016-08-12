#from datetime import datetime
#tstart = datetime.now()
import time
start_time = time.time()

# A method for computing contact sum 
class contact_sum:
	def __init__(self, u, t, r_cut, r_eff):
		"""Initializing the class variables that will be used later"""
 		self.u = u
 		self.t = t
 		self.r_cut = r_cut
 		self.r_eff = r_eff
 		
 	def get_contact_sum(self, i=1):
		"""A method for computing contact sum"""
		# Load necessary modules
		import MDAnalysis.lib.distances
		from MDAnalysis.lib.distances import distance_array
		import numpy as numpy
		# Select carbon atom 
		reference = self.u.select_atoms("resid %s and name %s" % (i, self.t[0]))
		# Select all atoms within 5.0A of carbon atom selection            
		close_contacts = self.u.select_atoms("(around %f resid %s and name %s) and (not resid %s) and (not type 102 103 104 109 111 112)" % (self.r_cut, i, self.t[0], i))      
		# Compute the distances between reference and close_contacts coordinates
		r_ij = distance_array(reference.coordinates(), close_contacts.coordinates())
		# Compute contact sum
		Ci = numpy.sum(numpy.exp(-1 * (r_ij / self.r_eff)))
		print Ci
		
	#def get_all_contact_sums(self):
		#for i in self.cutoffs_list:
			#Ci_list.append(self.get_contact_sum(i))
			
#tend = datetime.now()
#time = tend - tstart
#print time 
time = ("%s seconds" % (time.time() - start_time))
print time	


