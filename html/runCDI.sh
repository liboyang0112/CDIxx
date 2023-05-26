#!/bin/bash
shopt -s expand_aliases
source /home/boyang/softwares/Imaging/CDIxx/env.sh
cd $1
cdi_cu cdi.cfg
#rm $2 $3

