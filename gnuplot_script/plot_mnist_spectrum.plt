power = -3
mult = 10**(-power)
set xlabel "Normalized wave length"
set ylabel "Normalized Intensity"
stats "spectra_cont.txt" nooutput
set yrange [0:STATS_max_y*1.2*mult]
set label sprintf("{/Symbol \264}10^{%d}",power) at 1,STATS_max_y*1.23*mult
set term pdf size 3,3
set output "spectra.pdf"
plot "spectra_cont.txt" using 1:($2*mult) title "Continuous Spectrum" with lines lt 1 lw 4 lc rgb "blue",\
"spectra_comb.txt" using 1:($2*8) title "Comb-like Spectrum" with lines lt 1 lw 4 lc rgb "red"

