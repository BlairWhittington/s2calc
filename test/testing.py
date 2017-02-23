import numpy as numpy
import MDAnalysis

def test_alignment(PSF, DCD0, align = True, REF = "reference.pdb", DCD1 = "pool_align.dcd"):
    """ A method to test the alignment function"""
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
    
   