#!/bin/bash
source ../env.sh
export LD_LIBRARY_PATH=/usr/local/hpc_sdk/Linux_x86_64/23.7/math_libs/lib64:${LD_LIBRARY_PATH}
cd $1
echo processimg_cu $2 $3 $4 $5 $6 $7 $8
processimg_cu $2 $3 $4 $5 $6 $7 $8 2>&1
