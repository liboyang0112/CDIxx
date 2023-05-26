#!/bin/bash
source ../env.sh
cd $1
pulseGen_cu pulse.cfg
gnuplot ../gnuplot_script/plot_residual.plt
gnuplot ../gnuplot_script/plot_hj_spectrum.plt
