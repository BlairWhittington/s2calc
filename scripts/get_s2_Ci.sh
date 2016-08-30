#!/bin/bash 
rm -rf output.txt
r_cut="25.0"
rnas="1R7Z 1R7W 1XHP 1ZC5 2FDT 2JWV 2JYM 2L5Z 2LDT 2QH3"
#rnas="1R7Z 1R7W 1SCL 1XHP 1ZC5 2FDT 2JWV 2JYM 2L3E 2L5Z 2LDT 2QH3"
for rna in $rnas
do
    python scripts/use_s2.py -i ${rna} data/${rna}/${rna}.nomin.psf data/${rna}/pool.dcd | grep s2calc >> output/${r_cut}.txt
done

