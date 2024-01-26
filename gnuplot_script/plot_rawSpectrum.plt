#set title "Converging speed of different strategy"
set xlabel "Wave Length / nm"
set ylabel "Intensity"
#set yrange [1e-5:]
#set term png size 1000,500
#set output "rawspectrum.png"
set term pdf size 6,3
set output "rawspectrum.pdf"
plot "spectra.txt" title "Spectrum" with lines lt 1 lw 3 lc rgb "blue"

set logscale y
set output "rawspectrumlog.pdf"
replot
