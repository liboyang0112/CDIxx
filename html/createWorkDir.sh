#!/bin/bash
shopt -s expand_aliases
source ../env.sh
mkdir $1
cp ../config/cdi.cfg $1
cp ../config/pulse.cfg $1
cp ExperimentLog.txt $1

