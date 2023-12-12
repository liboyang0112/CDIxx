set xlabel "wavelength / [a.u.]"
set ylabel "Intensity, phase/PI [a.u.]"
set yrange [-1:1]
set key horiz
set term pdf size 6,3
set output "frog.pdf"
plot "spectrum_sim.txt" using 2:($6/3.1415926) tit "simulation: phase" with lines lc rgb "blue" lw 3 dt 4,\
"spectrum.txt" using 2:($6/3.1415926) tit "reconstruction: phase" with lines lc rgb "red" lw 3 dt 4,\
"spectrum_sim.txt" using 2:5 tit "simulation: intensity" with lines lc rgb "blue" lw 3,\
"spectrum.txt" using 2:5 tit "reconstruction: intensity" with lines lc rgb "red" lw 3,\
0 notitle with lines lc rgb "black" lw 1 dt 2

