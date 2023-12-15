set xlabel "wavelength / [a.u.]"
set ylabel "Intensity, phase/PI [a.u.]"
stats "spectrum_sim.txt" nooutput
lamb=650;
xmax = STATS_max_y*lamb
xmin = STATS_min_y*lamb
set xrange [xmin : xmax]
set yrange [-1:1.5]
set key horiz
set key width 1
set term pdf size 7,3
set output "spectPhase.pdf"
plot "spectrum_sim.txt" using ($2*lamb):($6/3.1415926) tit "simulation: phase" with lines lc rgb "blue" lw 3 dt 4,\
"spectrum.txt" using ($2*lamb):($6/3.1415926) tit "reconstruction: phase" with lines lc rgb "red" lw 3 dt 4,\
"spectrum_sim.txt" using ($2*lamb):5 tit "simulation: intensity" with lines lc rgb "blue" lw 3,\
"spectrum.txt" using ($2*lamb):5 tit "reconstruction: intensity" with lines lc rgb "red" lw 3,\
"spectrum_raw.txt" using ($1*lamb):2 tit "Illumination spectrum" with lines lc rgb "black" lw 3,\
0 notitle with lines lc rgb "black" lw 1 dt 2

