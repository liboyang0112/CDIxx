set xlabel "q [um^{-1}]"
set ylabel "PRTF"
set term pdf size 4,2
set output "prtf.pdf"
plot "prtf.txt" using ($1/2.56):($2) title "PRTF" with lines lt 1 lw 2 lc rgb "blue", exp(-1) title "1/e Threshold" with lines dt 4 lw 2 lc rgb "red"
