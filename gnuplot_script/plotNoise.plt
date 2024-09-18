set ylabel "SSIM"
set xlabel "Photon counts"
set ytics 0.05
set yrange [0.73:0.95]
set logscale x
set format x "10^{%L}"
set term pdf size 5,3 font ",16"
set output "noise.pdf"
set key height 1 offset 0,0.8 column 2
set xlabel offset 0,0.5
set ylabel offset 1.5,0

plot \
"err.txt" using 1:2 with lp title "Noise = 20" lw 3,\
"err.txt" using 1:4 with lp title "Noise = 50" lw 3,\
"err.txt" using 1:6 with lp title "Noise = 100" lw 3,\
"err.txt" using 1:8 with lp title "Noise = 200" lw 3

set ylabel "PSNR (dB)"
set output "noise_psnr.pdf"
set ytics 2
set yrange [19:27]
plot \
"err.txt" using 1:3 with lp title "Noise = 20" lw 3,\
"err.txt" using 1:5 with lp title "Noise = 50" lw 3,\
"err.txt" using 1:7 with lp title "Noise = 100" lw 3,\
"err.txt" using 1:9 with lp title "Noise = 200" lw 3
