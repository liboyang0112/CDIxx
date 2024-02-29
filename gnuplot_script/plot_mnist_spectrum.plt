file1="assim/spectrum.txt"
file2="combsim/spectrum_raw.txt"
file3="contsim/spectra_raw.txt"

power = -3
mult = 10**(-power)
set xlabel "Wavelength (nm)"
set ylabel "Normalized intensity (a.u.)"
stats file1 nooutput
max1 = STATS_max_y
stats file2 nooutput
max2 = STATS_max_y
stats file3 nooutput
max3 = STATS_max_y

set yrange [0:1]
set xrange [3:18]
#set label sprintf("{/Symbol \264}10^{%d}",power) at 1,STATS_max_y*1.25*mult
set term pdf size 6,2
set output "spectra.pdf"
plot file1 using 1:($2/max1*0.8) title "Experimental isolated attosecond pulse spectrum" with lines lt 1 lw 4 lc rgb "black",\
file2 using ($1*3):($2/max2*0.8) title "Comb-like spectrum" with lines lt 1 lw 4 lc rgb "red",\
file3 using ($1*3):($2/max3*0.8) title "Continuous spectrum" with lines lt 1 lw 4 lc rgb "blue",

