#!/bin/bash
source /home/boyang/myenv.sh
cat $2 | grep "	" > $1/spectrum.txt
cd $1
gnuplot $CDI_DIR/gnuplot_script/plot_rawSpectrum.plt
