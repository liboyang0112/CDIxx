#! /usr/bin/env bash
source myenv.sh
source selectGPU.sh
cd $1
echo processimg_ptycho_run $2 $3 $4 $5 $6 $7
processimg_ptycho_run $2 $3 $4 $5 $6 $7 2>&1
cp $2/scan.txt $3/.
