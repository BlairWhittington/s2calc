#!/bin/bash 
rm -rf output.txt
rnas="1R7Z 1R7Z"
for rna in $rnas
do
	python use_s2.py -i ${rna} ${rna}.nomin.psf pool.dcd | grep s2calc >> output.txt
done
