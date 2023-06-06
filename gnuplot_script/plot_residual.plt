set format y "10^{%L}"
#set title "Converging speed of different strategy"
set xlabel "Iteration"
set ylabel "Residual"
set logscale y
#set yrange [1e-5:]
set term png size 500,500
set output "residual_iter_pulseGen.png"
plot "residual.txt" title "Residual" with lines lt 1 lw 3 lc rgb "blue"
