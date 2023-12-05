#!/bin/bash
source /home/boyang/.bashrc
source ../env.sh
export LD_LIBRARY_PATH=/usr/local/hpc_sdk/Linux_x86_64/23.7/math_libs/lib64:${LD_LIBRARY_PATH}
cd $1
cdi_run cdi.cfg 2>&1
