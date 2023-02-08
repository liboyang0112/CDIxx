set timestamp
set title "spectrum comparison"
set xlabel "Normalized wave length"
set ylabel "Intensity"
set xrange [1:1.4]
set term pdf
set output "spectra.pdf"
plot "spectra.txt" title "Measured" with lines lt 1 lw 3 lc rgb "blue", "spectra_new.txt" title "reconstructed" with lines lt 1 lw 3 lc rgb "red"

