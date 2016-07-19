import numpy as numpy
import MDAnalysis 

class s2_calculator:
  """A method for Initializing s2 order parameter, and residue number"""
  def __init__(self, u, t):
    # initializing the class variable set by the user
    self.u = u
	self.t = t
    # initializing class variable that will be used later
    self.s2_list = []
    self.resid_list = []
    self.nframes = len(self.u.trajectory)
    
  def get_s2(self, i=1):
    # Set initial vector quantities equal to zero 
    x2 = 0
    y2 = 0
    z2 = 0
    xy = 0
    xz = 0
    yz = 0 
    # Make carbon atom selection
    sel1 = self.u.select_atoms("resid %s and name self.t[0]" % (i))
    # Check to see if a carbon atom has been selected
    if sel1.n_atoms != 1:
      print "error must select 1 carbon atom" 
    # Make hydrogen atom selection  
    sel2 = self.u.select_atoms("resid %s and name self.t[1]" % (i))
    # Check to see if a hydrogen atom has been selected
    if sel2.n_atoms != 1:
      print "error must select 1 hydrogen atom"
    # Loop over trajectory
    for ts in self.u.trajectory:
      sel1 = self.u.select_atoms("resid %s and name self.t[0]" % (i))
      if sel1.n_atoms != 1:
        print "error must select 1 carbon atom" 
      sel2 = self.u.select_atoms("resid %s and name self.t[1]" % (i))
      if sel2.n_atoms != 1:
        print "error must select 1 hydrogen atom" 
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
    for i in self.u.atoms.residues.resids:      
      self.s2_list.append(self.get_s2(i))
      self.resid_list.append(i)

  def get_scatterplot(self):
    """A method for plotting s2 vs. residue number
    Plot of s2 order parameters versus residue number 
    Create a scatterplot of s2 order parameters vs. residue number
    """
    from matplotlib import pyplot
    from numpy import arange
    import bisect 
    # Assign the variable x to list of resids
    x = self.resid_list
    # Assign the variable y to list of s2 order parameters
    y = self.s2_list
    # Plot the assigned variables as dots
    pyplot.plot(x,y,'.')
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
    

  