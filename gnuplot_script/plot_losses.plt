#set title "Converging speed of different strategy"
set xlabel "Epoch"
set ylabel "Loss [a.u.]"
#set yrange [1e-5:]
#set term png size 1000,500
#set output "rawspectrum.png"
set term pdf size 6,3
set output "loss.pdf"
set logscale y
set logscale x
plot "losses_bk.txt" using 1:2 title "Training" with lines lt 1 lw 3 lc rgb "blue", "losses_bk.txt" using 1:3 tit "Validation" with lines lt 1 lw 3 lc rgb "red"
