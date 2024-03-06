file1="assim/spectrum.txt"
file2="combsim/spectrum_raw.txt"
file3="contsim/spectra_raw.txt"

power = -3
mult = 10**(-power)
set xlabel "Wavelength (nm)"
set x2label "Wavelength (a.u.)"
set ylabel "Normalized intensity (a.u.)"
stats file1 nooutput
max1 = STATS_max_y
stats file2 nooutput
max2 = STATS_max_y
stats file3 nooutput
max3 = STATS_max_y

set yrange [0:1]
set xrange [2:12]
set x2tics 1,1
set xtics nomirror
#set label sprintf("{/Symbol \264}10^{%d}",power) at 1,STATS_max_y*1.25*mult
set term pdf size 7.5,2.5
set output "spectra.pdf"
plot file1 using 1:($2/max1*0.8) title "Experimental isolated attosecond pulse spectrum (nm)" with lines lt 1 dt 4 lw 4 lc rgb "black" axis x1y1,\
file2 using 1:($2/max2*0.8) title "Comb-like spectrum (a.u.)" with lines lt 1 lw 4 lc rgb "red" axis x2y1,\
file3 using 1:($2/max3*0.8) title "Continuous spectrum (a.u.)" with lines lt 1 lw 4 lc rgb "blue" axis x2y1,

