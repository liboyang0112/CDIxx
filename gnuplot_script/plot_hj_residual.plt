set format y "10^{%L}"
#set title "Converging speed of different strategy"
set xlabel "Iteration"
set ylabel "Residual"
set xrange [1:100]
set logscale y
set yrange [1e-5:]
set term pdf size 3,2
set output "residual.pdf"
plot "step_residual_1p4.txt" title "Gradiant" with lines lt 1 lw 3 lc rgb "blue",\
"momop8.txt" title "Momentum k=0.8" with lines lt 1 lw 3 lc rgb "red",\
"momop6.txt" title "Momentum k=0.6" with lines lt 1 lw 3 lc rgb "green"

