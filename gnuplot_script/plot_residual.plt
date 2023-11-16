set format y "10^{%L}"
#set title "Converging speed of different strategy"
set xlabel "Iteration"
set ylabel "Residual"
set logscale y
#set yrange [1e-5:]
set term pdf size 6,4
set output "residual_iter_pulseGen.pdf"
plot "residual.txt" title "Residual" with lines lt 1 lw 3 lc rgb "blue"
