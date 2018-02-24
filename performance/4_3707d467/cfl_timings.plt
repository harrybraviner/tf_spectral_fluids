set terminal png size 640, 480
set output "cfl_timings.png"

set linestyle 1 ps 1 pt 7
set linestyle 2 ps 1 pt 5
set style line 3 lw 1 lt rgb "gray"

set xlabel "Collocation points in each direction"
set ylabel "Wall-clock time per timestep"
set logscale x
set logscale y

set xtics (8, 16, 32, 64, 128, 256)

plot (x > 64 ? 7.8e-8*x**3 : 1.0/0.0) w l ls 3 title "N^3" ,\
(x > 64 ? (7.8e-8*1.111)*x**3 : 1.0/0.0) w l ls 3 title "" ,\
"cfl_timings.dat" using 1:2 ls 1 w p title "RK3, fixed timestep",\
"cfl_timings.dat" using 1:3 ls 2 w p title "RK3, CFL condition"

unset output
