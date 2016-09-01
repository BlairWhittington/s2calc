i=2
a = new.universe.selectAtoms(" resid %s and name O3\' "%(i-1)," resid %s and name P  "%(i)," resid %s and name O5\' "%(i)," resid %s and name C5\' "%(i))
a.dihedral.value()
 