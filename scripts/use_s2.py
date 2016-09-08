import s2_calculator
from s2_calculator import *
import contact_sum
from contact_sum import *
import Features
from Features import *
import MDAnalysis 
from itertools import repeat
from optparse import OptionParser
import warnings, sys

def main():
    #parse commandline
    usage = "usage: %prog [options] topology-file coordinate-file"
    parser = OptionParser(usage)
    parser.add_option("-i", "--id",dest="id",default="NONE",help="identification code included in the output [default: %default]")
    parser.add_option("-v", "--verbose",dest="verbose",metavar="BOOL",default=False,action="store_true",help="verbose output [default: %default]")
    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("incorrect number of arguments")
    else:
        #initialize variables from the commandline
        PSF=args[0]
        DCD0=args[1]

        u = initialize_universe(PSF, DCD0)
        
        t = [("C1'", "H1'")]
    	#t = [("C1'", "H1'"), ("C2", "H2"), ("C5", "H5"), ("C6", "H6"), ("C8", "H8"), ("N1", "H1"), ("N3", "H3")]
        
        nframes = len(u.trajectory)
        nresid = len(u.atoms.residues)    
   
        ### Use for s2_calculator ###
        s2 = s2_calculator(u, t)
        s2.get_all_s2()
        #s2.get_scatterplot()
        
        ### Use for contact_sum ###
        n=50
    	r_cut=10.0
        r_eff=5.0
        sum = contact_sum(u, t, n, r_cut, r_eff)
        sum.get_all_distances()
        print sum.distance_list
    	#sum.get_all_contact_sums()
    	
    	### Use for Features ###
    	new = Features(u)
    	new.unique_HBtable()
    	new.ts = 0
    	new.loadCoordinateDependentDataStructures()
    	new.residueLevelFeaturesCompute()
    	for key in u.atoms.residues.resids:
    		print key, new.system['%i'%key]
        
        ### Table of resids, s2 order parameters, Ci values, and atom selections ###
        #s2.s2_list = [ '%.8f' % i for i in s2.s2_list]
        #sum.Ci_list = ['%.8f' % i for i in sum.Ci_list]    
        #for i, v, f, lh, lp in zip(s2.resid_list, s2.s2_list, sum.Ci_list, s2.bond_vector_list_heavy,  s2.bond_vector_list_proton):
            #print "s2calc", options.id, i, v, f, lh, lp
  
if __name__ == "__main__":
    main()
    
    
