def test_alignment(PSF, DCD0, align = True, REF = "reference.pdb", DCD1 = "pool_align.dcd"):
    """ A method to test the alignment function"""
    from MDAnalysis.analysis.align import *
    # Load initial universe
    u = MDAnalysis.Universe(PSF, DCD0)
    # Test if alignment is requested; default is yes!
    # Align trajectory
    write_reference(u, REF, i=0) 
