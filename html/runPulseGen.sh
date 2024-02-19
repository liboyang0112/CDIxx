#!/bin/bash
source /home/boyang/myenv.sh
cd $1
echo $PWD
mono_run pulse.cfg 2>&1
gnuplot $CDI_DIR/gnuplot_script/plot_residual.plt
gnuplot $CDI_DIR/gnuplot_script/plot_hj_spectrum.plt
