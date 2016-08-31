import sys, getopt
import scipy.stats as stats
import numpy, string
from scipy.special import ellipk, ellipe
from math import pi, sin, cos, atan2, sqrt, pow, ceil, acos
from MDAnalysis.analysis.distances import distance_array
from multiprocessing import Queue, Process
from MDAnalysis.analysis.hbonds import HydrogenBondAnalysis
from MDAnalysis import Universe

class Features:
    """Features of S2 calculation"""
    def __init__(self, universe, hbondAtoms=False,stacticsLigandCalculate=False, radius=9999.0, rcRadius=9999.0, stackDistanceCutOff=6.0, stackingOverlapBuffer=1.0, stackAngularCutOff=30.0, hBondAngularCutoff=145, hBondDistanceCutoff=3.5): 
        """Initialize Features Class
        """
        self.universe = universe
        self.radius = radius 
        self.stackDistanceCutOff = stackDistanceCutOff
        self.stackingOverlapBuffer = stackingOverlapBuffer
        self.stackAngularCutOff = stackAngularCutOff
        self.hBondAngularCutoff = hBondAngularCutoff
        self.hBondDistanceCutoff = hBondDistanceCutoff
        self.segid = self.universe.residues[0].atoms[0].segid
        self.ligandSele = False
        self.hbondAtoms = hbondAtoms
        self.aromatics = []
        self.nuclevels = {"H1'":0,"H2'":1,"H3'":2,"H4'":3,"H5'":4,"H5''":5,"H2":6,"H5":7,"H6":8,"H8":9,"C1'":10,"C2'":11,"C3'":12,"C4'":13,"C5'":14,"C2":15,"C5":16,"C6":17,"C8":18}
        self.reslevels = {"G":0,"A":1,"C":2,"U":3,"T":4}
                        
        # rename atoms
        self.universe.selectAtoms("name H2'1").set_name("H2'")
        self.universe.selectAtoms("name H5'1").set_name("H5'")
        self.universe.selectAtoms("name H5'2").set_name("H5''")
        
        # rename nucleic acid residue names
        for i in  self.universe.residues.resnums:
            resname = self.universe.selectAtoms("resid %s"%(i)).resnames[0]
            if resname in ["G","GUA","RG","RG3","RG5","DG","DG3","DG5"]: self.universe.selectAtoms("resid %s"%(i)).set_resname("GUA")
            elif resname in ["A","ADE","RA","RA3","RA5","DA","DA3","DA5"]: self.universe.selectAtoms("resid %s"%(i)).set_resname("ADE")
            elif resname in ["U","URA","RU","RU3","RU5","DU","DU3","DU5"]: self.universe.selectAtoms("resid %s"%(i)).set_resname("URA")
            elif resname in ["C","CYT","RC","RC3","RC5","DC","DC3","DC5"]: self.universe.selectAtoms("resid %s"%(i)).set_resname("CYT")
            elif resname in ["T","THY","DT","DT3","DT5"]: self.universe.selectAtoms("resid %s"%(i)).set_resname("THY")
            elif resname in ["HIS","HSD","HSP","HSE","HID","HIE","HIP"]: self.universe.selectAtoms("resid %s"%(i)).set_resname("HIS")

        # prepare for hydrogen bonding detection
        nucleic_and_protein_donors = ['N7', 'N1', 'N2', 'N3', 'N4', 'N6', 'O1P', 'O2P', 'OP1','OP2', "O5'", "O4'", "O2'", "O3'",'HD1','HD2','HD21','HD22','HE','HE1','HE2','HE21','HE22','HG1','HH','HH11','HH12','HH21','HH22','HN','HN1','HN2','HNT','HT1','HT2','HT3','HZ1','HZ2','HZ3','N','ND1','ND2','NE','NE1','NE2','NH1','NH2','NT','NZ','OD2','OE2','OG','OG1','OH','SG']
        nucleic_and_protein_acceptors = ['O6', 'O1P', 'O2P', 'O2', 'O4', 'N9', 'N7', 'N4', 'N6', 'N1', 'N3', "O5'", "O4'", "O2'", "O3'",'C','CD','CG','CY','ND1','NE2','O','OD1','OD2','OE1','OE2','OG','OG1','OH','OT1','OT2','OY']

        # account for ligand if present
        ligdonors = ligacceptors = []
        
        sele = "not resname LIG"
        if(self.hbondAtoms):
            sele = "all"
            ligdonors,ligacceptors = self.loadLigandHbonds()
        donors = self.flattenList([nucleic_and_protein_donors,ligdonors])
        acceptors = self.flattenList([nucleic_and_protein_acceptors,ligacceptors]) 
        self.rnaRNA = HydrogenBondAnalysis(self.universe,sele,sele,donors=donors,acceptors=acceptors,distance=self.hBondDistanceCutoff,angle=self.hBondAngularCutoff,verbose=False, update_selection1=False, update_selection2=False, distance_type="heavy", detect_hydrogens="heuristic")
        self.rnaRNA.run()
        self.rnaRNA.timeseries
        self.rnaRNA.generate_table()
        self.HBtable = self.rnaRNA.table.astype([('time', '<i8'), ('donor_idx', '<i8'), ('acceptor_idx', '<i8'), ('donor_resnm', 'S4'), ('donor_resid', '<i8'), ('donor_atom', 'S4'), ('acceptor_resnm', 'S4'), ('acceptor_resid', '<i8'), ('acceptor_atom', 'S4'), ('distance', '<f8'), ('angle', '<f8')])
        self.universe.trajectory.rewind()
    
    def loadCoordinateDependentDataStructures(self):
        """ Loads data structures that depend on atomic coordinates. Function loads the distance 
            matrix, axes used to define group magnetic anisotropies, and aromatic rings structure info.
            Useful when looping over a trajectory (see larmorExtractor.py for example).
                :Arguments:
                  none 
                .. versionadded:: 1.0.0
        """
        self.coordinates = self.universe.selectAtoms("all").coordinates()
        self.distanceMatrix = distance_array(self.coordinates,self.coordinates)
        
    def residueLevelFeaturesCompute(self):
        """Residue Level Structure Feature. Function Compute the residue level features e.g. hydrogen bonding, torsions etc.
           Residue level features are stored in the dictionary: *system*
                :Arguments:
                  none
                .. versionadded:: 1.0.0
        """
        #loop over residues in system and populated system dictionary
        self.system = {}
        for i in self.universe.residues.resnums:
            resname = self.universe.selectAtoms("resid %s "%(i)).resnames[0]
            #populate system
            if resname in ["G","GUA","RG","RG3","RG5","DG","DG3","DG5"]:
                name = 'G'  
            elif resname in ["A","ADE","RA","RA3","RA5","DA","DA3","DA5"]:
                name = 'A'
            elif resname in ["C","CYT","RC","RC3","RC5","DC","DC3","DC5"]:
                name = 'C'
            elif resname in ["U","URA","RU","RU3","RU5","DU","DU3","DU5"]:
                name = 'U'
            self.system['%s'%i] = {}
            self.system['%s'%i]['resname'] = name
            resnamecode = [ str(x) for x in numpy.zeros(len(self.reslevels.keys()),dtype="int")]
            resnamecode[self.reslevels['%s'%name]] = '1'
            self.system['%s'%i]['resnamecode'] = resnamecode
            if self.isNucleic(resname):
                #hydrogen bonding
                self.system['%s'%i]['hbond'] = self.hBondEnergyCompute(i)
                #torsions
                #self.system['%s'%i]['tors'] = self.tors(i)
                #stacking interactions
                self.system['%s'%i]['stacking'] = self.stackingEnergyCompute(i)

    def unique_HBtable(self):
        """ function to return only unique HBtable entries """
        x, y = [list(self.HBtable.field(1)),list(self.HBtable.field(2))]
        z = []
        for a, b in zip(x,y):
          z.append([a,b]) 
        a = numpy.array(z)
        order = numpy.lexsort(a.T)
        a = a[order]
        self.HBtable = self.HBtable[order] 
        diff = numpy.diff(a, axis=0)
        ui = numpy.ones(len(a), 'bool')
        ui[1:] = (diff != 0).any(axis=1) 
        self.HBtable = self.HBtable[ui] 

    def hBondEnergyCompute(self,resnum):
        """h-bond energy: simple counts number of h-bonds residue resnum is involved in
        :Arguments:
        *resnum*
        :counts number of H-bonds this residue is involved in
        *hBondDistanceCutoff*
        :distance cutoff between h-bond donor and acceptor ( must be <= *cutoff*)
        *hBondAngularCutoff*
        :angles cutoff between Donor--H--Acceptor (must be >= *cutoff*)
        .. versionadded:: 1.0.0
        """
        hEnergy_base_base = hEnergy_base_back = hEnergy_back_back = 0        
        backbone = [ "O1P", "O2P","O5'","O4'","H2'1", "O2'", "HO'2","O3'"]                
        #print "Comparing Time from HBond and Index: %s %s"%(self.HBtable['time'],self.ts)   
        #self.rnaRNA.table.astype([('time', '<i8'), ('donor_idx', '<i8'), ('acceptor_idx', '<i8'), ('donor_resnm', 'S4'), ('donor_resid', '<i8'), ('donor_atom', 'S4'), ('acceptor_resnm', 'S4'), ('acceptor_resid', '<i8'), ('acceptor_atom', 'S4'), ('distance', '<f8'), ('angle', '<f8')])     
        for r in self.HBtable[numpy.where(self.HBtable['time']==self.ts)]:            
            if resnum in [r[4],r[7]] and r[4] != r[7]:
                hbDist = r[9]
                hbAngle = r[10]
                atom1 = r[1]-1
                atom2 = r[2]-1
                atom1=self.universe.selectAtoms("all").names[atom1]
                atom2=self.universe.selectAtoms("all").names[atom2]                
                if (hbAngle >= self.hBondAngularCutoff and hbDist <= self.hBondDistanceCutoff):
                  if atom1 not in backbone and atom2 not in backbone:
                      hEnergy_base_base += 1
                  if atom1 in backbone and atom2 not in backbone:
                      hEnergy_base_back += 1
                  if atom2  in backbone and atom1 not in backbone:
                      hEnergy_base_back += 1
                  if atom1 in backbone and atom2 in backbone:
                      hEnergy_back_back += 1                    
        #print hEnergy_base_base, hEnergy_base_back, hEnergy_back_back
        return [hEnergy_base_base, hEnergy_base_back, hEnergy_back_back]

    def stackingEnergyCompute(self,ref_residue):
        """stacking energy of *ref_residue* with other bases in rna
        takes the atom group for set of atoms in RNA base and returns the normalized normal to the plane formed by these atoms
        :Arguments:
        *universe*
        :class:`~MDAnalysis.core.AtomGroup.Universe`
        *ref_residue*
        :calculate stacking energy for this residue
        *radius*
        :search for stacking partner with *radius* of the center of mass (COM) of *ref_residue*
        *stackDistanceCutOff*
        :distance cutoff between COM of *ref_residue* and potential stacking partner ( must be <= *cutoff*)
        *stackAngularCutOff*
        :angles between normals of *ref_residue* and potential stacking partner planes must in range [-*angular_cutoff*,*angular_cutoff*]
        *stackingOverlapBuffer*
        : projected atoms of potential stacking partners must < radius of base plane + *radius_buffer*
        .. versionadded:: 1.0.0
        """
        reference = self.bases_only(self.universe.selectAtoms("resid %s"%(ref_residue)))
        comparison = self.bases_only(self.universe.selectAtoms("(around %f resid %s) and (not (resid %s))"%(self.radius,ref_residue,ref_residue)))
        norm1 = self.base_plane_normal(reference)
        base_radius = numpy.max(distance_array(self.residue_COM(reference),reference.coordinates()))+self.stackingOverlapBuffer
        stacking_score = 0.0
        stacking_score_ligand = 0.0
        for i in comparison.residues.resnums:
            #get resid coordinates
            check = self.bases_only(self.universe.selectAtoms("resid %s"%(i)))
            com_distances = distance_array(self.residue_COM(reference),self.residue_COM(check))
            if (com_distances < self.stackDistanceCutOff):
                    norm2 = self.base_plane_normal(check)
                    angle = numpy.rad2deg(numpy.arccos(numpy.dot(norm1,norm2)))
                    angle = min(180-angle,angle)                    
                    if ((angle>=-1*self.stackAngularCutOff) and (angle<=self.stackAngularCutOff)):
                         for coord in check.coordinates():
                             test = coord - self.residue_COM(reference)
                             test = self.norm_vec(test-self.dot(test,norm1)*norm1)
                             if (test < base_radius):                                 
                                 stacking_score += 1
                                 break
        #process ligand contribution if any
        if (self.ligandSele and self.stackingCalculate):
            comparison = self.universe.selectAtoms("(around %f resid %s) and (resname LIG)"%(self.radius,ref_residue))
            if comparison.n_atoms > 0:
                if (self.aromatics and True) or False:
                    for aromatic in self.aromatics.keys():
                        ligsele = self.aromatics[aromatic]
                        check = self.universe.selectAtoms(ligsele)
                        com_distances = distance_array(self.residue_COM(reference),self.residue_COM(check))
                        if (com_distances <= self.stackDistanceCutOff):
                                norm2 = self.base_plane_normal(check)
                                angle = numpy.rad2deg(numpy.arccos(self.dot(norm1,norm2)))
                                angle = min(180-angle,angle)
                                if ((angle>=-1*self.stackAngularCutOff) and (angle<=self.stackAngularCutOff)):
                                     for coord in comparison.coordinates():
                                         test = coord - self.residue_COM(reference)
                                         test = self.norm_vec(test-self.dot(test,norm1)*norm1)
                                         if (test < base_radius+2):
                                             stacking_score_ligand += 1
                                             break
        return stacking_score+stacking_score_ligand
 
    def tors(self,i):   
        """Torsions. Computes alpha, beta, gamma, delta, epsilon, zeta, chi and nu0, nu1, nu2, nu3, nu4
        :Arguments:
          *i*: residue for which to compute dihedrals
             resid of the base
        .. versionadded:: 1.0.0
        """
        a = self.universe.selectAtoms(" resid %s and name O3\' "%(i-1)," resid %s and name P  "%(i)," resid %s and name O5\' "%(i)," resid %s and name C5\' "%(i))
        b = self.universe.selectAtoms(" resid %s and name P    "%(i)," resid %s and name O5\' "%(i)," resid %s and name C5\' "%(i)," resid %s and name C4\' "%(i))
        g = self.universe.selectAtoms(" resid %s and name O5\' "%(i)," resid %s and name C5\' "%(i)," resid %s and name C4\' "%(i)," resid %s and name C3\' "%(i))
        d = self.universe.selectAtoms(" resid %s and name C5\' "%(i)," resid %s and name C4\' "%(i)," resid %s and name C3\' "%(i)," resid %s and name O3\' "%(i))
        e = self.universe.selectAtoms(" resid %s and name C4\' "%(i)," resid %s and name C3\' "%(i)," resid %s and name O3\' "%(i)," resid %s and name P    "%(i+1))
        z = self.universe.selectAtoms(" resid %s and name C3\' "%(i)," resid %s and name O3\' "%(i)," resid %s and name P    "%(i+1)," resid %s and name O5\' "%(i+1))
        n0 = self.universe.selectAtoms("resid %s  and name  C1\' "%(i)," resid %s  and name  C2\' "%(i)," resid %s  and name  C3\' "%(i)," resid %s  and name C4\' "%(i))
        n1 = self.universe.selectAtoms("resid %s  and name  C2\' "%(i)," resid %s  and name  C3\' "%(i)," resid %s  and name  C4\' "%(i)," resid %s  and name O4\' "%(i))
        n2 = self.universe.selectAtoms("resid %s  and name  C3\' "%(i)," resid %s  and name  C4\' "%(i)," resid %s  and name  O4\' "%(i)," resid %s  and name C1\' "%(i))
        n3 = self.universe.selectAtoms("resid %s  and name  C4\' "%(i)," resid %s  and name  O4\' "%(i)," resid %s  and name  C1\' "%(i)," resid %s  and name C2\' "%(i))
        n4 = self.universe.selectAtoms("resid %s  and name  O4\' "%(i)," resid %s  and name  C1\' "%(i)," resid %s  and name  C2\' "%(i)," resid %s  and name C3\' "%(i))
        try:
            c = self.universe.selectAtoms(" resid %s and name O4\' "%(i)," resid %s and name C1\' "%(i)," resid %s and name N1 "%(i)," resid %s and name C2  "%(i))
        except:
            c = self.universe.selectAtoms(" resid %s and name O4\' "%(i)," resid %s and name C1\' "%(i)," resid %s and name N9 "%(i)," resid %s and name C4  "%(i))
        alpha = beta = gamma = delta = epsilon = zeta = chi = nu0 = nu1 = nu2 = nu3 = nu4 =9999.9
        if (a.n_atoms == 4): alpha = a.dihedral.value()
        if (b.n_atoms == 4): beta = b.dihedral.value()
        if (g.n_atoms == 4): gamma = g.dihedral.value()
        if (d.n_atoms == 4): delta = d.dihedral.value()
        if (e.n_atoms == 4): epsilon = e.dihedral.value()
        if (z.n_atoms == 4): zeta = z.dihedral.value()
        if (c.n_atoms == 4): chi = c.dihedral.value()

        if (n0.n_atoms == 4): nu0 = n0.dihedral.value()
        if (n1.n_atoms == 4): nu1 = n1.dihedral.value()
        if (n2.n_atoms == 4): nu2 = n2.dihedral.value()
        if (n3.n_atoms == 4): nu3 = n3.dihedral.value()
        if (n4.n_atoms == 4): nu4 = n4.dihedral.value()
        
        #if alpha < 0 :
        #    alpha = alpha+360
        #if beta < 0 :
        #    beta = beta+360
        #if gamma < 0 :
        #    gamma = gamma+360
        #if epsilon < 0:
        #    epsilon = epsilon+360
        #if zeta < 0 :
        #    zeta = zeta+360
        #if chi < 0 :
        #    chi = chi+360
        #if nu0 < 0 :
        #    nu0 = nu0+360
        #if nu1 < 0 :
        #    nu1 = nu1+360
        #if nu2 < 0 :
        #    nu2 = nu2+360
        #if nu3 < 0 :
        #    nu3 = nu3+360
        #if nu4 < 0 :
        #    nu4 = nu4+360
        ##return [cos(numpy.deg2rad(alpha)), cos(numpy.deg2rad(beta)), cos(numpy.deg2rad(gamma)), cos(numpy.deg2rad(delta)), cos(numpy.deg2rad(epsilon)), cos(numpy.deg2rad(zeta)), cos(numpy.deg2rad(chi)), cos(numpy.deg2rad(nu0)), cos(numpy.deg2rad(nu1)), cos(numpy.deg2rad(nu2)), cos(numpy.deg2rad(nu3)), cos(numpy.deg2rad(nu4)), cos(numpy.deg2rad(chi)),sin(numpy.deg2rad(alpha)), sin(numpy.deg2rad(beta)), sin(numpy.deg2rad(gamma)), sin(numpy.deg2rad(delta)), sin(numpy.deg2rad(epsilon)), sin(numpy.deg2rad(zeta)), sin(numpy.deg2rad(chi)), sin(numpy.deg2rad(nu0)), sin(numpy.deg2rad(nu1)), sin(numpy.deg2rad(nu2)), sin(numpy.deg2rad(nu3)), sin(numpy.deg2rad(nu4))]
        #return [cos(numpy.deg2rad(alpha)), cos(numpy.deg2rad(beta)), cos(numpy.deg2rad(gamma)), cos(numpy.deg2rad(delta)), cos(numpy.deg2rad(epsilon)), cos(numpy.deg2rad(zeta)), cos(numpy.deg2rad(chi)), cos(numpy.deg2rad(nu0)), cos(numpy.deg2rad(nu1)), cos(numpy.deg2rad(nu2)), cos(numpy.deg2rad(nu3)), cos(numpy.deg2rad(nu4)), cos(numpy.deg2rad(chi)),sin(numpy.deg2rad(alpha)), sin(numpy.deg2rad(beta)), sin(numpy.deg2rad(gamma)), sin(numpy.deg2rad(delta)), sin(numpy.deg2rad(epsilon)), sin(numpy.deg2rad(zeta)), sin(numpy.deg2rad(chi)), sin(numpy.deg2rad(nu0)), sin(numpy.deg2rad(nu1)), sin(numpy.deg2rad(nu2)), sin(numpy.deg2rad(nu3)), sin(numpy.deg2rad(nu4))]
        return [alpha, beta, gamma, delta, epsilon, zeta, chi, nu0, nu1, nu2, nu3, nu4]
                    
    def loadLigandSelections(self):
        """Load Ligand Selection. Add path to ligand selection file to Larmor
                :Arguments:
                  none
                .. versionadded:: 1.0.0
        """
        if (self.ligandSele):
            selections = {}
            lines = open(self.ligandSele,"r")
            for i, line in enumerate(lines):
                self.RCLigIntensity.append(float(line.split(":")[0]))
                selections[i] = line.split(":")[1]
                print ("Ligand Selection %s: %s")%(i,selections[i])
            self.aromatics = selections

    def loadLigandHbonds(self):
        """Load Hbond Atoms.
                :Arguments:
                  none
                .. versionadded:: 1.0.0
        """
        if (self.hbondAtoms):
            donor = []
            acceptors = []
            lines = open(self.hbondAtoms,"r")
            for i, line in enumerate(lines):
                  tag = line.split(":")[0]
                  if tag in ["acceptors"]:
                        acceptors = line.split(":")[1].split()
                  if tag in ["donors"]:
                        donors = line.split(":")[1].split()
            return donors,acceptors
                    
    def distanceCompute(self,ref_residue,atom):
        """Calculate distance to nearest P"""
        close = []
        for neighborAtom in self.neighborAtoms:
            reference = self.universe.selectAtoms("resid %s and name %s"%(ref_residue,atom))
            close_contacts = self.universe.selectAtoms("(name %s) and (not resid %s)"%(neighborAtom,ref_residue))
            close.append(numpy.min(distance_array(reference.coordinates(),close_contacts.coordinates())))
        return close
          
    def get_bonded_atoms(self,atomname,resname,bonds_list={}):
        """return selection string for atoms bonded to query *atomname*
        take *atomname* and *resname*~(only needed for C1', C2, C5, C6, C8)
        and return a selection string of atoms bonded to *atomname*. This is
        only valid for nucleic acids. Further only valid for H1',H2',H3',
        H4',H5',H5'',C1',C2',C3',C4',C5',C2,C5,C6,C8 atoms. Dictionary is store
        in global variable name **bonds**
        :Arguments:
          *bondlist*
                 :dictionary with *atomname*:selection pairs 
          *atomname*
                 :atom name of query atom
          *resname*
                 :residue name for query atom   
        .. versionadded:: 1.0.0
        """    
        valid = ["H1'","H2'","H3'","H4'","H5'","H5''","H2","H5","H6","H8","C1'","C2'","C3'","C4'","C5'","C2","C5","C6","C8"]
        if atomname not in valid: raise Exception("atomname must be one of these %s"%(valid))   
        if atomname not in ["C1'","C2'","C3'","C4'","C5'","C2","C5","C6","C8"]:
            return bonds_list[atomname]
        else:
            if resname=="None": 
                raise Exception("must specify a rename for carbon %s"%(atomname))  
            elif resname in ["G","GUA","RG","RG3","RG5","DG","DG3","DG5"]:
                resname = "G"
                return bonds_list[atomname][resname]
            elif resname in ["A","ADE","RA","RA3","RA5","DA","DA3","DA5"]:
                resname = "A"
                return bonds_list[atomname][resname]
            elif resname in ["C","CYT","RC","RC3","RC5","DC","DC3","DC5"]:
                resname = "C"
                return bonds_list[atomname][resname]
            elif resname in ["U","URA","RU","RU3","RU5","DU","DU3","DU5"]:
                resname = "U"
                return bonds_list[atomname][resname]
            elif resname in ["T","THY","DT","DT3","DT5"]:
                resname = "T"
                return bonds_list[atomname][resname]
        
    def isNucleic(self,resname):
        return resname in ["G","GUA","RG","RG3","RG5","DG","DG3","DG5","A","ADE","RA","RA3","RA5","DA","DA3","DA5","U","URA","RU","RU3","RU5","DU","DU3","DU5","C","CYT","RC","RC3","RC5","DC","DC3","DC5","T","THY","DT","DT3","DT5"]
       
    #utility functions
    def residue_COM(self,atomGroup):
        """calculates the center of mass of residues in given *atomGroup*.     
        The center of mass for all residues in the input *atomGroup* is determined and returned as numpy.array    
        :Arguments:
          *atomGroup* 
             :class:`~MDAnalysis.core.AtomGroup.Universe.AtomGroup` 
        .. versionadded:: 1.0.0
        """    
        list = []
        for i in atomGroup.residues.resnums:
            list.append(atomGroup.selectAtoms("resid %s"%(i)).centerOfMass())
        return numpy.asarray(list,dtype=numpy.float32)
    
    def bases_only(self,atomGroup):
        """retains only base atoms in *atomGroup*.     
        takes nucleic acid atom group and return only atoms in the base    
        :Arguments:
          *atomGroup* 
             :class:`~MDAnalysis.core.AtomGroup.Universe.AtomGroup` 
        .. versionadded:: 1.0.0
        """    
        return atomGroup.selectAtoms("not (name C1' or name C2' or name C3' or name C4' or name C5' or name H* or name P* or name O*P or protein)")

    def bases_only_protein(self,atomGroup):
        """retains only base atoms in *atomGroup*.     
        takes nucleic acid atom group and return only atoms in the base    
        :Arguments:
          *atomGroup* 
             :class:`~MDAnalysis.core.AtomGroup.Universe.AtomGroup` 
        .. versionadded:: 1.0.0
        """    
        return atomGroup.selectAtoms("(name CG or name CD1 or name CE1 or name CZ or name CE2 or name CD2 or name CG or name CD2 or name CE2 or name NE1 or name CD1 or name CE3 or name CZ3 or name CH2 or name CZ2 or name ND1 or name CE1 or name NE2 or name CD2)")
      
    def fitPlaneSVD(self,XYZ):
        """SVD solution to fit plane based an set of points
        """    
        [rows,cols] = XYZ.shape
        # Set up constraint equations of the form  AB = 0,
        # where B is a column vector of the plane coefficients
        # in the form b(1)*X + b(2)*Y +b(3)*Z + b(4) = 0.
        p = (numpy.ones((rows,1)))
        AB = numpy.hstack([XYZ,p])
        [u, d, v] = numpy.linalg.svd(AB,0)        
        B = v[3,:];                    # Solution is last column of v.
        nn = numpy.linalg.norm(B[0:3])
        B = B / nn
        return B[0:3]
    
    def base_plane_normal(self,atomGroup):
        """normal vector for plane RNA form by base defined in *atomGroup*.     
        takes the atom group for set of atoms in RNA base and returns the normalized normal to the plane formed by these atoms
        :Arguments:
          *atomGroup* 
             :class:`~MDAnalysis.core.AtomGroup.Universe.AtomGroup` 
        .. versionadded:: 1.0.0
        """    
        #check if number of residues is get than one
        if(atomGroup.n_residues>1): raise Exception("Only valid for single residue selections")     
        return fitPlaneSVD(atomGroup.coordinates())
    
    def base_plane_normal(self,atomGroup):
        """normal vector for plane RNA form by base defined in *atomGroup*.     
        takes the atom group for set of atoms in RNA base and returns the normalized normal to the plane formed by these atoms
        :Arguments:
          *atomGroup* 
             :class:`~MDAnalysis.core.AtomGroup.Universe.AtomGroup` 
        .. versionadded:: 1.0.0
        """    
        #check if number of residues is get than one
        if(atomGroup.n_residues>1): raise Exception("Only valid for single residue selections") 
        a = atomGroup.coordinates()[2]-atomGroup.coordinates()[0]
        b = atomGroup.coordinates()[4]-atomGroup.coordinates()[0]
        return self.normalize_vec(numpy.cross(b,a))
    
    def base_plane_angle(self,universe,i,j):
        norm1 = base_plane_normal(bases_only(universe.selectAtoms("resid %s"%(i))))
        norm2 = base_plane_normal(bases_only(universe.selectAtoms("resid %s"%(j))))
        angle = numpy.rad2deg(numpy.arccos(numpy.dot(norm1,norm2)))
        angle = min(180-angle,angle)
        return angle
    
    def base_plane_distance(self,universe,ref_residue,i):
        """distance between *ref_residue* base plane and center of mass of *i* base    
        takes as input the base atoms of *ref_residue* and *i* and calculates the minimum distance
        between plane defined by *ref_residue* and center of mass of *i*
        :Arguments:
          *universe* 
             :class:`~MDAnalysis.core.AtomGroup.Universe` 
          *ref_residue*
             :reference base  *ref_residue*
          *i*
             :comparison base
        .. versionadded:: 1.0.0
        """    
        norm = base_plane_normal(bases_only(universe.selectAtoms("resid %s"%(ref_residue))))
        com1 =  residue_COM(bases_only(universe.selectAtoms("resid %s"%(ref_residue))))
        com2 =  residue_COM(bases_only(universe.selectAtoms("resid %s"%(i))))
        f = com2-com1
        return numpy.abs(numpy.dot(f,norm))[0]

    def dot(self,x,y):
        """calculate the dot product"""
        return numpy.sum(x*y)
        
    def normalize_vec(self,v): 
        """normalize a vector"""
        return (v/numpy.sqrt(numpy.sum(v*v)))

    def norm_vec(self,v):
        """get vector norm""" 
        return (numpy.sqrt(numpy.sum(v*v)))     

    def projection_onto_plane(self,point,norm):
        #Calculates the projection of "point" o on to the ring plane defined by normal "norm"
        d = self.dot(point,norm)/self.norm_vec(norm)# or can be written as dot(point,normalize_vec(norm))
        p = d*self.normalize_vec(norm)
        return point - p

    def areaN(self,a,b,n):
        #calculates the area New of a triangle given vectors r1,r2, normal
        crossProd = numpy.cross(b,self.normalize_vec(n))
        return .5*self.dot(a,crossProd)

    def area(a,b,c):
        #calculates the area of a triangle given 3 points a,b,c
        #return 0.5 * norm_vec( numpy.cross( b-a, c-a ) )
        if norm_vec(numpy.cross( b-a, c-b ))==0:
            d = 0
            # print  d
            #else:
            # print dot(normalize_vec(numpy.cross( b-a, c-b )),normalize_vec(norm))
        if dot(normalize_vec(numpy.cross( b-a, c-b )),norm)>0:   #==1
            d = .5
        if dot(normalize_vec(numpy.cross( b-a, c-b )),norm)<0:
            d = -.5           
        if norm_vec(norm)==0:
            print ("error")
        #d = .5
        return d*norm_vec(numpy.cross( b-a, c-a)) 

    def flattenList(self,l):
        """ Flattens a nested list (2-levels)
        """
        return [item for sublist in l for item in sublist]

    def removeDuplicates(self,seq):
        """ removes duplicates from a list
        """
        seen = set()
        seen_add = seen.add
        return [ x for x in seq if x not in seen and not seen_add(x)]
    
    def worker(self,nums,out_q,processor):
        """ The worker function, invoked in a process. 'nums' is a
            list of numbers to factor. The results are placed in
            a dictionary that's pushed to a queue.
        """
        outdict = {}
        outdict[processor] = self.calculateResidueFeatures(nums)
        out_q.put(outdict)

    def getFeatures(self):
        nums = self.shiftResidues
        nprocs = self.nprocs
        resultdict = {}
        if (nprocs == 1):
            resultdict[0] = self.calculateResidueFeatures(nums)
            return resultdict
        else:
            chunksize = int(ceil(len(nums) / float(nprocs)))
            procs = []
            out_q = Queue()
            for i in range(nprocs):        
                p = Process(target=self.worker,args=(nums[chunksize * i:chunksize * (i + 1)],out_q,i,))
                procs.append(p)
                p.start()
                # print "started %s"%(i)
            # Collect all results into a single result dict. We know how many dicts
            # with results to expect.
            for i in range(nprocs):
                resultdict.update(out_q.get())
                # print "collecting results %s"%(i)
            # Wait for all worker processes to finish
            for p in procs:
                p.join()
                # print "waiting for all process to finish %s"%(p)
            return resultdict

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
