rij = distance between heavy atom of bond vector i and atom j
reff =  determines effective length scale (5 Angstroms)
rcut = cutoff distance 
contact sum (Ci) = sum(e ^ (-1 * (rij / reff)) for each contact distance for all js within 
rcut of i 

- calculate distance between carbon atom of bond vector i and atom j
- already calculated bond vector i - vecCH - need to access carbon atom of bond vector
- need to loop through all atoms within rcut
- for each ij pair, calculate (e ^ (-1 * (rij /reff))) 
- sum over these values to get contact sum 

- since heavy atom of bond vector i is carbon selection, can i be set to sel1? YES
- create a pdb that only contains atoms within rcut of i? NOT NEEDED
- loop through all atoms 

- create two versions of function:
	- one that loops over all atoms in trajectory, checks to see if resid of both atoms 
	selected are the same, if resid is different check distance, and if rij is within
	rcut - calculate contact sum 
	 
- "around 5 sel1": selects all atoms, b, around 5 A of carbon atom selection, a 
- for all atoms, b, within 5 A of a, calculate distance, r
	if r is less than rcut, compute contact value, c
	sum over all values of c for each atom, b, within rcut of atom a 

- since "around 5 sel1" will select ALL atoms, need to access each atom to calculate distance
#######################################################################################

self.ci_list = []

def get_contact_sum(self, i=1):
	""" A method for computing the contact sum"""
	self.reff = 
	self.rcut = 5
	for ts in u.trajectory:
		a = self.u.select_atoms("resid %s and name %s" % (i, self.t[0]))
		self.sel2 = self.u.select_atoms(around 5 self.sel1 "resid %s and name %s" % (i))
		b = self.sel2
		if self.sel1.atoms.residues.resids == self.sel2.atoms.residues.resids:
			return False
		else: 
			return True
		r = self.sel1.center_of_mass() - self.sel2.center_of_mass()
		rab = math.sqrt((r[0] ** 2) + (r[1] ** 2) + (r[2] ** 2 ))
		if rab < self.rcut:
			import math
			f = (-1 * (rab / self.reff))
			c = math.exp(f)
			c += c
			return c
		
	for i in range(self.sel2.atoms.residues.resids):
		self.ci_list.append(get_contact_sum(i))
	
			
def get_contact_sum(self, i=1):
	""" A method for computing contact sum"""
	self.reff = 
	self.rcut = 
	for ts in u.trajectory:
		self.sel1 = self.u.select_atoms("resid %s and name %s" % (i, self.t[0]))			
		self.sel2 = self.u.select_atoms(around 5 self.sel1)		
		for i in self.sel2:
			if self.sel1.atoms.residues.resids == self.sel2.atoms.residues.resids:
				return False
			else:
				return True 
			r = self.sel1.center_of_mass() - self.sel2.center_of_mass()
			rij = math.sqrt((r[0] ** 2) + (r[1] ** 2) + (r[2] ** 2))
			if rij < self.rcut:
				f = (-1 * (rij / self.reff))
				c = math.exp(f)
				c += c
				return c


	b = self.sel1.center_of_mass()
	d = self.sel2.center_of_mass()
	self.rij = math.sqrt(((b[0] - d[0]) ** 2) + ((b[1] - d[1]) ** 2) + ((b[2] - d[2]) ** 2))
	self.rij = self.sel2.center_of_mass() - self.sel1.center_of_mass()