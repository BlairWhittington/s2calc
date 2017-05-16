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
        
        t = [("C1'", "H1'"), ("C2", "H2"), ("C5", "H5"), ("C6", "H6"), ("C8", "H8"), ("N1", "H1"), ("N3", "H3")]
         
        nframes = len(u.trajectory)
        nresid = len(u.atoms.residues)    
   
        ### Use for s2_calculator ###
        s2 = s2_calculator(u, t)
        s2.get_all_s2()
        #print s2.s2_list
        #s2.get_scatterplot()
        
        ### Use for contact_sum ###
        n=50
        r_cut=10.0
        r_eff=5.0
        sum = contact_sum(u, t, n, r_cut, r_eff)
        sum.get_all_distances()
        sum.get_resnames()
        
        ### Use for Features### 
        new = Features(u)
        new.unique_HBtable()
        new.ts = 0
        new.loadCoordinateDependentDataStructures()     
        new.residueLevelFeaturesCompute()
        
        ### Table of keys, s2 order parameters, distances, stacking, tors, hbond ### 
        for lh, r, i, v in zip(s2.bond_vector_list_heavy, sum.resnames, s2.resid_list, s2.s2_list):
            if v != -1:
            	out = "%s %s %s %s %s %s %s %s %s" %(options.id, v, convert_name_to_code(lh), i, convert_resname_to_code(r), new.system['%s'%i]['stacking'], new.system['%s'%i]['tors'], new.system['%s'%i]['hbond'], sum.distance_list[i, lh])
            	out = ''.join(s for s in out if ord(s)>31 and ord(s)<126 and s not in '[]' and s not in ',')
            	print out      
     
if __name__ == "__main__":
    main()
    
    
