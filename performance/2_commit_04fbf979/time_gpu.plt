set terminal png size 640, 480
set output "time_gpu.png"

set linestyle 1 ps 1 pt 7
set linestyle 2 ps 1 pt 5
set style line 3 lw 1 lt rgb "gray"

set xlabel "Collocation points in each direction"
set ylabel "Wall-clock time per timestep"
set logscale x
set logscale y

set xtics (8, 16, 32, 64, 128, 256)

plot "time_gpu.dat" using 1:2 w l ls 1 title "no anti-aliasing",\
"time_gpu.dat" using 1:3 w l ls 2 title "with anti aliasing"

unset output
