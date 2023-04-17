power = -3
mult = 10**(-power)

stats "spectra_raw.txt" nooutput
set yrange [0:STATS_max_y*1.2*mult]
set label sprintf("{/Symbol \264}10^{%d}",power) at 1,STATS_max_y*1.23*mult

#set title "Spectrum"
set xlabel "Normalized wave length"
set ylabel "Normalized Intensity"
set xrange [1:1.4]
set yrange [0:]
set term pdf size 3,3
set output "spectra.pdf"
plot "spectra_raw.txt" using 1:($2*mult) title "Spectrum" with lines lt 1 lw 2 lc rgb "blue",\
"spectra.txt" using 1:($2*mult) title "Points taken" with points pt 7 ps 0.3 lc rgb "red"

