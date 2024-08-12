
set xrange [0:200]
set yrange [0:1]
set term pdf size 4,2
set output "resolution.pdf"
set xlabel "y [{/symbol m}m]"
set ylabel "Intensity [a.u.]"

plot "linedata.txt" using ($1*1.6-20):($2/6) with lines lw 3 lc rgb "blue" title "Ti:Sapphire laser", "linedata.txt" using ($1*1.6-20):($3/6) with lines lw 3 dt 4 lc rgb "red" title "He:Ne laser"
