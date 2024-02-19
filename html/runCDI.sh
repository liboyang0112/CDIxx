#!/bin/bash
source /home/boyang/myenv.sh
cd $1
cdi_run cdi.cfg 2>&1
