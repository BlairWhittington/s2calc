resnum = 2
hEnergy_base_base = hEnergy_base_back = hEnergy_back_back = 0        
backbone = [ "O1P", "O2P","O5'","O4'","H2'1", "O2'", "HO'2","O3'"]                
for r in new.HBtable[numpy.where(new.HBtable['time']==new.ts)]:
  if resnum in [r[4],r[7]]:
      hbDist = r[9]
      hbAngle = r[10]
      atom1 = r[1]-1
      atom2 = r[2]-1
      atom1=new.universe.selectAtoms("all").names[atom1]
      atom2=new.universe.selectAtoms("all").names[atom2]                
      if (hbAngle >= new.hBondAngularCutoff and hbDist <= new.hBondDistanceCutoff):
        if atom1 not in backbone and atom2 not in backbone:
            hEnergy_base_base += 1
        if atom1 in backbone and atom2 not in backbone:
            hEnergy_base_back += 1
        if atom2  in backbone and atom1 not in backbone:
            hEnergy_base_back += 1
        if atom1 in backbone and atom2 in backbone:
            hEnergy_back_back += 1                    
        print hEnergy_base_base, hEnergy_base_back, hEnergy_back_back, r[4], r[7], atom1, atom2






