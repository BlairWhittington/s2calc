# A method for computing contact sum 
class contact_sum:
    def __init__(self, u, t, r_cut, r_eff):
        """Initializing the class variables that will be used later"""
        self.u = u
        self.t = t
        self.r_cut = r_cut
        self.r_eff = r_eff
        self.Ci_list = []
        
    def get_contact_sum_one(self, i=1):
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
        return Ci
        
    def get_all_contact_sums(self):
        """A method for looping over residues to compute all contact sums"""
        for i in self.u.atoms.residues.resids:
            self.Ci_list.append(self.get_contact_sum_one(i)) 
                
    def get_contact_sum_two(self):
        """A function for computing the contact sum"""
        # Load necessary modules
        import MDAnalysis.lib.distances
        from MDAnalysis.lib.distances import distance_array
        import numpy as numpy
        # Create a matrix that contains only heavy atom distances
        heavy_atoms = self.u.select_atoms("not type 102 103 104 109 111 112")
        dismat = distance_array(heavy_atoms.coordinates(), heavy_atoms.coordinates())
        # Get carbon atom selection index
        dist = dismat[:, 4]
        # Use index to slice heavy atom distance matrix
        dist = dist[20:len(dismat)]
        # Create a list that contains distances within 5.0A of carbon atom selection
        dist_list = []
        for i in dist:
            if i != 0.0 and i < self.r_cut:
                dist_list.append(i)
        # Compute contact sum 
        r_ij = numpy.array(dist_list)
        Ci = numpy.sum(numpy.exp(-1 * (r_ij / self.r_eff)))
        print Ci

    



