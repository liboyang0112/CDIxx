#!/bin/bash
source myenv.sh
cd $1
echo processimg_run $2 $3 $4 $5 $6 $7 $8
processimg_run $2 $3 $4 $5 $6 $7 $8 2>&1
