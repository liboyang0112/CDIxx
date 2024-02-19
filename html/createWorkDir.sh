#!/bin/bash
shopt -s expand_aliases
source /home/boyang/myenv.sh
mkdir $1
cp default/cdi.cfg $1
cp default/pulse.cfg $1
cp default/ExperimentLog.txt $1

