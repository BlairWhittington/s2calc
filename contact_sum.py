self.ci_list = []

def get_contact_sum(self, i=1):
	""" A method for computing contact sum"""
	self.reff = 5
	self.rcut = 5
	for ts in u.trajectory:
		self.sel1 = self.u.select_atoms("resid %s and name %s" % (i, self.t[0]))			
		self.sel2 = self.u.select_atoms(around 5 self.sel1)		
		for l in self.sel2:
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
				
	for i in range(self.sel2.atoms.residues.resids):
		self.ci_list.append(get_contact_sum(i))
		
		
		
import numpy.linalg
from numpy.linalg import norm

numpy.linalg.norm(a - b) for a, b in zip(self.sel1, self.sel2(i))