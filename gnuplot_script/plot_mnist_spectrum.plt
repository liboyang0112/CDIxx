file1="assim/spectrum.txt"
file2="combsim/spectrum_raw.txt"
file3="contsim/spectra_raw.txt"

maxr = 0.6
set xlabel "Wavelength (nm)"
set x2label "Wavelength (a.u.)"
set ylabel "Intensity (a.u.)" offset 1.5
stats file1 nooutput
max1 = STATS_max_y/maxr
stats file2 nooutput
max2 = STATS_max_y/maxr
stats file3 nooutput
max3 = STATS_max_y/maxr

set yrange [0:1]
set xrange [2:12]
set x2tics 1,1 offset 0, -0.3
set xtics 1,1 offset 0, 0.3
set xtics nomirror
set xlabel offset 0,0.8
set x2label offset 0,-0.8
set term pdf size 4,3 font ",16"
set output "spectra.pdf"
plot file1 using 1:($2/max1) title "Experimental IAP spectrum (nm)" with lines lt 1 dt 4 lw 4 lc rgb "black" axis x1y1,\
file2 using 1:($2/max2) title "Comb-like spectrum (a.u.)" with lines lt 1 lw 4 lc rgb "red" axis x2y1,\
file3 using 1:($2/max3) title "Continuous spectrum (a.u.)" with lines lt 1 lw 4 lc rgb "blue" axis x2y1,

