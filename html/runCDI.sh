#!/bin/bash
source myenv.sh
source selectGPU.sh
cd $1
cdi_run cdi.cfg 2>&1
