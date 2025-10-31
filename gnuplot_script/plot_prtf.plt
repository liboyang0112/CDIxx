set xlabel "q [µm^{-1}]"
set ylabel "PRTF"
set term pdf size 4,2

set output "/dev/null"
# Step 1: Plot with watchpoint to find intersection
plot "prtf.txt" using ($1/2.56/5):($2) title "PRTF" with lines lt 1 lw 2 lc rgb "blue" \
    watch y=exp(-1) watch y=0.5, \
     exp(-1) title "y=1/e" with lines dt 4 lw 2 lc rgb "red"

# Step 2: Retrieve the intersection point if it exists
x_intersect = NaN
x_intersect_2 = NaN
if (exists("WATCH_1") && |WATCH_1| > 0) {
    x_intersect = real(WATCH_1[1])
}

if (exists("WATCH_2") && |WATCH_2| > 0) {
    x_intersect_2 = real(WATCH_2[1])
}

# Step 3: Calculate resolution in nanometers
resolution_nm = (x_intersect > 0) ? (1e3 / x_intersect) : -1
resolution_nm_2 = (x_intersect_2 > 0) ? (1e3 / x_intersect_2) : -1

# Step 4: Add label to the plot if we found a valid intersection
if (x_intersect > 0) {
    # Format the label text
    set arrow 1 from x_intersect, graph 0 to x_intersect, graph 1 nohead lw 2 lc rgb "black" dt 2
    label_text_q0 = sprintf("q_{0} = %.2f µm^{-1}", x_intersect)
    set label 2 at x_intersect, graph 1.08 center label_text_q0 tc rgb "black" font ",10"
    label_text = sprintf("R = 1/q_0 = %.1f nm \\@ 1/e threshold\nR = 1/q_1 = %.1f nm \\@ 0.5 threshold", resolution_nm, resolution_nm_2)
    
    # Place the label on the graph (remove 'text')
    set label 1 at screen 0.25, 0.15 center label_text tc rgb 0x00cc0000 font ",10"
}
else {
    set label 1 "No 1/e crossing found" at screen 0.5, 0.5 center tc rgb "red" font ",10"
}

# Final replot to include the label
set output "prtf.pdf"
replot
