#!/bin/bash
source ../env.sh
export LD_LIBRARY_PATH=/usr/local/hpc_sdk/Linux_x86_64/23.7/math_libs/lib64:${LD_LIBRARY_PATH}
cd $1
echo $PWD
pulseGen_cu pulse.cfg 2>&1
gnuplot $CDI_DIR/gnuplot_script/plot_residual.plt
gnuplot $CDI_DIR/gnuplot_script/plot_hj_spectrum.plt
