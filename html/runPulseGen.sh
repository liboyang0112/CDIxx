#!/bin/bash
source ../env.sh
cd $1
echo $PWD
pulseGen_cu pulse.cfg
gnuplot $CDI_DIR/gnuplot_script/plot_residual.plt
gnuplot $CDI_DIR/gnuplot_script/plot_hj_spectrum.plt
