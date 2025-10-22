#!/bin/bash
source myenv.sh
cd $1
gnuplot $CDI_DIR/gnuplot_script/plot_prtf.plt
