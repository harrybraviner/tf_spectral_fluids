set terminal png size 640, 480
set output "time_per_step.png"

set linestyle 1 ps 1 pt 7
set linestyle 2 ps 1 pt 5
set style line 3 lw 1 lt rgb "gray"

set xlabel "Collocation points in each direction"
set ylabel "Wall-clock time per timestep"
set logscale x
set logscale y

set xtics (8, 16, 32, 64, 128, 256)

plot (0.1/(50**3))*x**3 w l ls 3 title "N^3" ,\
"time_per_step.dat" using 1:3 ls 1 w p title "CPU",\
"time_per_step.dat" using 1:4 ls 2 w p title "GPU"

unset output
