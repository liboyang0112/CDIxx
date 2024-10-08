
#file_raw = "spectrum.txt"
#file_raw = "spectra_raw.txt"
file_raw = "spectTccd.txt"
file = "spectrum_solved.txt"
#file_raw = "spectra_raw.txt"
#file = "spectra.txt"
set xrange [500:1000]

stats file_raw nooutput
ratioraw = 1. / STATS_mean_y
max = 1.2*STATS_max_y/STATS_mean_y

stats file nooutput 
ratio = 1. / STATS_mean_y
if(max < 1.2*STATS_max_y/STATS_mean_y) max = 1.2*STATS_max_y/STATS_mean_y
#set title "Spectrum"
set xlabel "Wavelength (nm)"
set ylabel "Intensity (a.u.)"
set yrange [0:max]
set xrange [450:1000]
set ytics 2
set term pdf size 5,2 font ",16"
set output "spectra.pdf"
#set term png size 1000,500
#set output "spectra.png"
plot file_raw using 1:($2*ratioraw) title "Measured" with lines lt 1 lw 2 lc rgb "blue",\
file using 1:($2*ratio) title "Reconstructed" with lp lt 1 lw 2 lc rgb "red" pt 7 ps 0.3 
#file using 1:($2*ratio) title "Points taken" with histograms
#file using 1:($2*ratio) title "Points taken" with lines lt 1 lw 3 lc rgb "red"

