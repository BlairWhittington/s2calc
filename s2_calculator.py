import numpy as numpy
import MDAnalysis 

class s2_calculator:
    """A method for initializing s2 order parameter, and residue number"""
    def __init__(self, u, t):
        """Initializing the class variable that will be used later"""
        self.u = u
        self.t = t
        self.s2_list = []
        self.resid_list = []
        self.carbon_list = []
        self.hydrogen_list = []
        self.nframes = len(self.u.trajectory)
	    self.nresid = len(self.u.atoms.residues)

	def super_impose_frames(self):
		"""A method to superimpose frame in a trajectory to a common, reference frame (e.g., the first frame in the trajectory)"""
		from MDAnalysis import *
		from MDAnalysis.analysis.align import *
		import MDAnalysis.analysis.align
		u = self.u
		l = self.u.select_atoms("all")
		with MDAnalysis.Writer("l.dcd", l.n_atoms) as W:
			ref = u.trajectory[0]
			for ts in u.trajectory:
				W.write(l)
		ref = self.l
		mobile = self.u
		alignto(mobile, ref, select="all", mass_weighted=True)
		
    def select_bond_vector(self, i):
		"""A method for selecting a single bond vector"""
		for i in t:
			self.sel1 = self.u.select_atoms("resid %s and name %s" % (i, self.t[0]))  
			self.sel2 = self.u.select_atoms("resid %s and name %s" % (i, self.t[1])) 
			if self.sel1.n_atoms != 1 and self.sel2.n_atoms != 1:
				print "selection not present in structure"
			
			b = self.select_bond_vector(i)
			if b < 0:
				print "error"
			else: #if this function is correct, indent functions below
						
    def get_s2(self, i=1):
        """A method for computing s2 order parameter for one residue"""
        # Set initial vector quantities equal to zero 
        x2 = 0
        y2 = 0
        z2 = 0
        xy = 0
        xz = 0
        yz = 0
        
        # Make carbon atom selection
        #sel1 = self.u.select_atoms("resid %s and name %s" % (i, self.t[0]))
        # Check to see if a carbon atom has been selected
        #if sel1.n_atoms != 1:
            #print "error must select 1 carbon atom" 
        # Make hydrogen atom selection  
        #sel2 = self.u.select_atoms("resid %s and name %s" % (i, self.t[1]))
        # Check to see if a hydrogen atom has been selected
        #if sel2.n_atoms != 1:
            #print "error must select 1 hydrogen atom" 
        # Loop over trajectory
        for ts in self.u.trajectory:
            # Define vector CH
            vecCH = sel1.center_of_mass() - sel2.center_of_mass()
            import numpy
            vecCH = self.normalize_vec(vecCH) 
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
        x2 = x2 / self.nframes
        y2 = y2 / self.nframes
        z2 = z2 / self.nframes
        xy = xy / self.nframes
        xz = xz / self.nframes
        yz = yz / self.nframes
        # Calculate s2 
        s2 = (1.5 * ((x2 ** 2) + (y2 ** 2) + (z2 ** 2))) + (3 * ((xy ** 2) + (xz ** 2) + (yz ** 2))) - 0.5
        return s2
            
    # A method for computing s2 order parameters for all residues
    def get_all_s2(self):
        """A method for iterating over all residues to compute all s2 order parameters"""
        for i in self.u.atoms.residues.resids:      
            self.s2_list.append(self.get_s2(i))
            self.resid_list.append(i)

    def get_scatterplot(self):
        """A method for plotting s2 vs. residue number"""
        from matplotlib import pyplot
        from numpy import arange
        import bisect 
        # Assign the variable x to list of resids
        x = self.resid_list
        # Assign the variable y to list of s2 order parameters
        y = self.s2_list
        # Plot the assigned variables as dots connected by lines
        pyplot.plot(x,y, marker='o', linestyle='-', color='b')
        # Label the x-axis Residue Number
        pyplot.xlabel('Residue Number')
        # Label the y-axis s2 Order Parameter
        pyplot.ylabel('s2 Order Parameter')
        # Show the plot 
        pyplot.show()
                  
    def get_table(self):
        """A method for listing s2 order parameters by resids and bond vectors in a table"""
        from astropy.table import Table, Column
        import numpy as np
        from itertools import repeat
        # Define a column resid that stores information from self.resid_list
        resid = self.resid_list
        # Define a column s2 that stores information from self.s2_list
        s2 = self.s2_list
        # Define carbon as a list containing carbon atom selection repeated once for each residue
        self.carbon_list.extend(repeat(self.t[0], self.nresid))
        carbon = self.carbon_list
        # Define hydrogen as a list containing hydrogen atom selection repeated once for each residue 
        self.hydrogen_list.extend(repeat(self.t[1], self.nresid))
        hydrogen = self.hydrogen_list
        # Create a table containing resid, s2 order parameter, carbon and hydrogen atoms selection information 
        s = Table([resid, s2, carbon, hydrogen], names = ("resid", "s2", "carbon_selection", "hydrogen_selection"))
        s
        
    # Normalize vectorCH  
    def norm_vec(self, v):
        """get vector norm"""
        return (numpy.sqrt(numpy.sum(v * v)))
  
    def normalize_vec(self, v):
        """normalize vector"""
        vecCH = (v / self.norm_vec(v))
        return vecCH
    

  