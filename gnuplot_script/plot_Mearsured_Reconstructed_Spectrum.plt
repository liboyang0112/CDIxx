set title "spectrum comparison"
set xlabel "Normalized wave length"
set ylabel "Intensity"
set format y "10^{%L}"
set xrange [1:1.4]
set logscale y
set yrange [1e-4:]
set term pdf size 3,3
set output "spectra.pdf"
plot "spectra.txt" title "Measured" with lines lt 1 lw 3 lc rgb "blue", "spectra_new.txt" title "reconstructed" with lines lt 1 lw 3 lc rgb "red"

