set format y "10^{%L}"
#set title "Converging speed of different strategy"
set xlabel "Iteration"
set ylabel "Residual"
set xrange [1:100]
set logscale y
set yrange [1e-5:]
set term pdf size 3,3
set output "residual.pdf"
plot "step_residual_1p4.txt" title "Gradiant" with lines lt 1 lw 3 lc rgb "blue", "mom_residual.txt" title "Momentum f=0.1" with lines lt 1 lw 3 lc rgb "red", "residual.txt" title "Momentum f=0.2" with lines lt 1 lw 3 lc rgb "green"

