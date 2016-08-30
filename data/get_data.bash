#!/bin/bash
module load mmtsb
mkdir -p /home/afrankz/tmp/collect_data
cd /home/afrankz/tmp/collect_data
datahome=/home/zdw/PROJECT/00_RESULTS

rnas="jxs k41 k8s 1r7w 1r7z 1xhp 1zc5 2fdt 2jwv 2jym 2l3e 2l5z 2ldt 2qh3"
for rna in ${rnas}
do
	mkdir -p ${rna}
	pdb2traj.pl -out ${rna}/pool.dcd ${datahome}/${rna}_c36/pdb/*frame*pdb
	cp ${datahome}/${rna}_c36/${rna}_nowat.psf ${rna}/reference.psf
	cp ${datahome}/${rna}_c36/pdb/${rna}_frame1.pdb ${rna}/reference.pdb
done

