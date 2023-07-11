
file_raw = "spectrum.txt"
file = "spectrum_solved.txt"
#file_raw = "spectra_raw.txt"
#file = "spectra.txt"
set xrange [500:1000]

stats file_raw
ratioraw = 1. / STATS_mean_y
max = 1.2*STATS_max_y/STATS_mean_y

stats file
ratio = 1. / STATS_mean_y
if(max < 1.2*STATS_max_y/STATS_mean_y) max = 1.2*STATS_max_y/STATS_mean_y
#set title "Spectrum"
set xlabel "Normalized wave length"
set ylabel "Normalized Intensity"
set yrange [0:max]
#set term pdf size 3,3
#set output "spectra.pdf"
set term png size 1000,500
set output "spectra.png"
plot file_raw using 1:($2*ratioraw) title "Spectrum" with lines lt 1 lw 2 lc rgb "blue",\
file using 1:($2*ratio) title "Points taken" with lp lt 1 lw 2 pt 7 ps 2. lc rgb "red"
#file using 1:($2*ratio) title "Points taken" with lines lt 1 lw 3 lc rgb "red"

