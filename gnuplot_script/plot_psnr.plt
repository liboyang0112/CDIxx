inputfile="psnr_25.dat"
inputfile2="psnr_50.dat"
inputfile3="psnr_200.dat"
set ylabel "SSIM"
set xlabel "Photon counts"
set ytics 0.1
set logscale x
set format x "10^{%L}"
set term pdf size 6,3 font "Times New Roman, 29"
set output "noise.pdf"
set key height 1 offset 0,0.8 column 2
set xlabel offset 0,0.5
set ylabel offset 1.5,0

Shadecolor1 = "#80FF0000"
Shadecolor2 = "#800000FF"
Shadecolor3 = "#8000FF00"

set xrange [4e6:2e9]
set yrange [0.55:0.95]
plot \
inputfile using 1:($4+$5):($4-$5) with filledcurve fc rgb Shadecolor1 title "Bkg = 25",\
inputfile2 using 1:($4+$5):($4-$5) with filledcurve fc rgb Shadecolor2 title "Bkg = 50",\
inputfile3 using 1:4:5 with yerr lw 3 pt 0 lc rgb "#0000CC00" title "Bkg = 200",\
#'' using 1:4 smooth mcspline lw 2 title "Background=25",
#'' using 1:4 smooth mcspline lw 2 title "Background=50",
#'' using 1:4 smooth mcspline lw 2 title "Background=200"
#inputfile using 1:4 with lp title "Background = 50" lw 3,\
#inputfile using 1:6 with lp title "Background = 100" lw 3,\
#inputfile using 1:8 with lp title "Background = 200" lw 3

set ylabel "PSNR (dB)"
set output "noise_psnr.pdf"
set ytics 2
#set yrange [19:27]
plot \
inputfile using 1:2:3 with yerr title "Background = 20" lw 3
#inputfile using 1:5 with lp title "Background = 50" lw 3,\
#inputfile using 1:7 with lp title "Background = 100" lw 3,\
#inputfile using 1:9 with lp title "Background = 200" lw 3
