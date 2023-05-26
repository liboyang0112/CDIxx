#!/bin/bash
shopt -s expand_aliases
source /home/boyang/softwares/Imaging/CDIxx/env.sh
mkdir $1
cp $CDI_DIR/config/cdi.cfg $1
cp $CDI_DIR/config/pulse.cfg $1
#rm $2 $3

