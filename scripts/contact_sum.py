import MDAnalysis.lib.distances
from MDAnalysis.lib.distances import distance_array
import numpy as np

class contact_sum:
    def __init__(self, u, t, n, r_cut, r_eff):
        """Initializing the class variables that will be used later"""
        self.u = u
        self.t = t
        self.n = n
        self.r_cut = r_cut
        self.r_eff = r_eff
        self.distance_list = {}
        self.keys = []
        self.Ci_list = []
                     
    def get_contacts(self, i , l):
        """A method for computing contact sum where i and l are the resid and bond vector 
        of interest, respectively"""
        # Assign reference to first atom selection
        self.reference = self.u.select_atoms("resid %s and name %s" % (i, l[0]))                
        # Check to see if selections exist
        if self.reference.n_atoms != 1:
            return False
        else:
            # Select all atoms within r_cut of first atom selection    
            self.close_contacts = self.u.select_atoms("(around %f resid %s and name %s) and ((not resid %s) and (not name H*))" % (self.r_cut, i, l[0], i))
            return True
            
    def get_distances(self, i, l):
        """A method for sorting and selecting the first 50 distances
        where i and l are the resid and bond vector of interest, respectively"""
        selection = self.get_contacts(i, l)
        if selection:
            # Get distance array
            global r_ij
            r_ij = distance_array(self.reference.coordinates(), self.close_contacts.coordinates())
            # Sort distances in array
            r_ij.sort()
            # Pad array with zeros if size of array is less than given n
            while r_ij.size <= self.n:
                r_ij = np.append(r_ij, 0)
            r_ij = r_ij[0:self.n]  
            # Simply slice distance array if size of array is larger than given n 
            if r_ij.size > self.n:
                r_ij = r_ij[0][0:self.n]
        return r_ij
    
    def get_all_distances(self):
        """A method for looping over residues to obtain all distances"""
        for l in self.t:
            for i in self.u.atoms.residues.resids:
            	key = i, l[0] 
                #key = str(i)+":"+str(l[0])  
                self.distance_list[key] = self.get_distances(i, l)
    
    #def get_contact_sum(self, i, l):
        #"""A method for checking if selection exists"""
        #Ci = -1 
        #selection = self.get_contacts(i, l)
        #if selection:
            #r_ij = distance_array(self.reference.coordinates(), self.close_contacts.coordinates())
            #Ci = numpy.sum(numpy.exp(-1 * (r_ij / self.r_eff))) 
        #return Ci
    
    #def get_all_contact_sums(self):
        #"""A method for looping over residues to compute all contact sums"""
        #for l in self.t:
            #for i in self.u.atoms.residues.resids:
                #self.Ci_list.append(self.get_contact_sum(i, l)) 

