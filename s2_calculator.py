import numpy as numpy
import MDAnalysis 

def initialize_universe(PSF, DCD0, align = True, REF = "reference.pdb", DCD1 = "pool_align.dcd"):
    """ A method to set up universe for S2 calculation;
    returns a universe that may or may not be aligned to some reference"""
    from MDAnalysis.analysis.align import *
    # Load initial universe
    u = MDAnalysis.Universe(PSF, DCD0)
    # Test if alignment is requested; default is yes!
    # Align trajectory
    write_reference(u, REF, i=0) 
    # Reads in reference pdb
    ref = MDAnalysis.Universe(PSF, REF)
    trj = u
    rms_fit_trj(trj, ref, filename=DCD1)
    # Read in new universe with aligned trajectory file
    u = MDAnalysis.Universe(PSF, DCD1)
    return u

def write_reference(u, output_name, i=0):
    """ This function writes out a specific frame in a universe"""
    u.trajectory[i]
    u.atoms.write(output_name)
    
class s2_calculator:
    """A method for initializing s2 order parameter, and residue number"""
    def __init__(self, u, t):
        """Initializing the class variables that will be used later"""
        self.u = u
        self.t = t
        self.s2_list = []
        self.resid_list = []
        self.nframes = len(self.u.trajectory)
        self.nresid = len(self.u.atoms.residues)
    
    def select_bond_vector(self, i=1):
        """A method for selecting a single bond vector"""
        self.sel1 = self.u.select_atoms("resid %s and name %s" % (i, self.t[0]))  
        self.sel2 = self.u.select_atoms("resid %s and name %s" % (i, self.t[1])) 
        if self.sel1.n_atoms != 1 or self.sel2.n_atoms != 1:
            return False
        else:
            return True
            
    def get_s2(self, i=1):
        """A method for computing s2 order parameter for one residue"""
        # Set initial vector quantities equal to zero 
        x2 = 0
        y2 = 0
        z2 = 0
        xy = 0
        xz = 0
        yz = 0   
        # Check selection        
        s2 = -1
        sele = self.select_bond_vector(i)        
        if sele:
            # Loop over trajectory
            for ts in self.u.trajectory:
                # Define vector CH
                vecCH = self.sel1.center_of_mass() - self.sel2.center_of_mass()
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
            
    def get_all_s2(self):
        """A method for iterating over all residues to compute all s2 order parameters"""
        for i in self.u.atoms.residues.resids:
        #for i in self.u.select_atoms("name %s" % (self.t[1])).residues.resids:     
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
        # Title the scatterplot
        pyplot.title('s2 Order Parameters vs. Resids')
        # Plot the assigned variables as dots connected by lines
        pyplot.plot(x,y, marker='o', linestyle='-', color='b')
        # Label the x-axis Residue Number
        pyplot.xlabel('Residue Number')
        # Label the y-axis s2 Order Parameter
        pyplot.ylabel('s2 Order Parameter')
        # Show the plot 
        pyplot.show()
        
    # Normalize vectorCH  
    def norm_vec(self, v):
        """get vector norm"""
        return (numpy.sqrt(numpy.sum(v * v)))
  
    def normalize_vec(self, v):
        """normalize vector"""
        vecCH = (v / self.norm_vec(v))
        return vecCH
    
    
