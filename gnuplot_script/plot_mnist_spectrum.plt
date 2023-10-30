power = -3
mult = 10**(-power)
set xlabel "Normalized wavelength (a.u.)"
set ylabel "Normalized intensity (a.u.)"
stats "spectra_cont.txt" nooutput
set yrange [0:1]
#set label sprintf("{/Symbol \264}10^{%d}",power) at 1,STATS_max_y*1.25*mult
set term pdf size 6,2
set output "spectra.pdf"
plot "spectra_cont.txt" using 1:($2*mult/3) title "Continuous Spectrum" with lines lt 1 lw 4 lc rgb "blue",\
"spectra_comb.txt" using 1:($2*8/3) title "Comb-like Spectrum" with lines lt 1 lw 4 lc rgb "red"

