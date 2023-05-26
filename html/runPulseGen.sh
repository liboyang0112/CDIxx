#!/bin/bash
shopt -s expand_aliases
source /home/boyang/softwares/Imaging/CDIxx/env.sh
cd $1
pulseGen_cu pulse.cfg
gnuplot /home/boyang/softwares/Imaging/CDIxx/gnuplot_script/plot_residual.plt
gnuplot /home/boyang/softwares/Imaging/CDIxx/gnuplot_script/plot_hj_spectrum.plt
#rm $2 $3

