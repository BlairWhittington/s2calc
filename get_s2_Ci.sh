#!/bin/bash 
r_cuts="10.0"
#r_cuts="10.0 15.0 20.0 25.0"
#rnas="1xhp 1r7w 1r7z 1zc5 2fdt 2jwv 2jym 2l3e 2l5z 2qh3 jxs k8s k41"
rnas="1xhp 1r7w 1r7z 1zc5 2fdt 2jwv 2jym 2l5z 2qh3 jxs k8s k41"
#rnas="2l5z 2qh3 jxs k8s k41"
for r_cut in $r_cuts
do
rm -rf output/${r_cut}.txt
	for rna in $rnas
	do 
		python scripts/use_s2.py -i ${rna} data/${rna}/reference.psf data/${rna}/pool.dcd | tee -a output/${r_cut}.txt
		#exit
	done
done
