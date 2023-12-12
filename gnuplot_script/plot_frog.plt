set xlabel "t [a.u.]"
set ylabel "Intensity / phase/PI [a.u.]"
set yrange [-1:1.5]
set key horiz
set term pdf size 6,3
set output "frog.pdf"
plot "input.txt" using 1:($5/3.1415926) tit "simulation: phase" with lines lc rgb "blue" lw 3 dt 4,\
"output.txt" using 1:($5/3.1415926) tit "reconstruction: phase" with lines lc rgb "red" lw 3 dt 4,\
"input.txt" using 1:4 tit "simulation: intensity" with lines lc rgb "blue" lw 3,\
"output.txt" using 1:4 tit "reconstruction: intensity" with lines lc rgb "red" lw 3

set xlabel "Frequency [a.u.]"
set ylabel "Intensity / phase/PI [a.u.]"
set term pdf size 6,3
set output "frog_spectrum.pdf"
plot "inputSpect.txt" using 1:($5/3.1415926) with lines tit "simulation: phase" lc rgb "blue" lw 3 dt 4,\
"outputSpect.txt" using 1:($5/3.1415926) tit "reconstruction: phase" with lines lc rgb "red" lw 3 dt 4,\
"inputSpect.txt" using 1:4 tit "simulation: intensity" with lines lc rgb "blue" lw 3,\
"outputSpect.txt" using 1:4 tit "reconstruction: intensity" with lines lc rgb "red" lw 3
