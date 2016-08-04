def get_contact_sum(self, i=1):
	""" A method for computing contact sum"""
	import MDAnalysis.lib.distances
	from MDAnalysis.lib.distances import distance_array 
	self.r_eff = 5.0
	self.r_cut = 5.0
	reference = self.u.select_atoms("resid %s and name %s" % (i, self.t[0]))			
	close_contacts = self.u.select_atoms("(around 5.0 resid %s and name %s) and (not resid %s) and (not type hydrogen)" % (i, self.t[0], i))		
	r_ij = distance_array(reference.coordinates(), close_contacts.coordinates())
	Ci = numpy.sum(numpy.exp(-1 * (r_ij / r_eff)))
	return Ci
	
	
	
	
- create an array that contains distances between all atom distance coordinates of u
- create a function that can identify the index of carbon atom selection
- use the carbon atom index to slice the array and get distances
- loop over the array 
- accumulate sums 
	
### Alternative method for computing contact sum
def get_carbon_atom_index(self):
	""" An Alternative method for computing contact sum"""
	dismat = distance_array(self.u.atoms.coordinates(), self.u.atoms.coordinates())
	# Find carbon atom selection index 
	for i in self.u.select_atoms("name C1'"):
		print i
	# Use carbon atom index to slice array
	dismat[8]
	# Get distances
	carbon_dist = distance_array(reference.coordinates(),close_contacts.coordinates())
	# Get contact sum
	Ci = numpy.sum(numpy.exp(-1 * (carbon_dist / r_eff)))
	return Ci
	
	
	
	
	
	
	for i, j in enumerate(dismat):
		if j == "C1'":
			print i
			
	print([atom.index for atom in u.atoms if atom.element.symbol is "C1'"])
	
	for i in u.select_atoms("resid %s and name C1'" % (i)):
		print i

	u.select_atoms("name C1'").coordinates()
	
	for i in dismat[8]:
		if i < 5.0:
			print i
			
	for i in dismat[8]:
         if i < 5.0:
   					array.append(i)