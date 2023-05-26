#!/bin/bash
shopt -s expand_aliases
source /home/boyang/softwares/Imaging/CDIxx/env.sh
cd $1
stacker_run $2 $3
#rm $2 $3

