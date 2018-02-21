set terminal png size 640, 480
set output "rk3.png"

set linestyle 1 ps 1 pt 7
set linestyle 2 ps 1 pt 5
set style line 3 lw 1 lt rgb "gray"

set xlabel "Collocation points in each direction"
set ylabel "Wall-clock time per timestep"
set logscale x
set logscale y

set xtics (8, 16, 32, 64, 128, 256)

plot (x > 64 ? (9.8e-3/(50**3))*x**3 : 1.0/0.0) w l ls 3 title "N^3" ,\
(x > 64 ? (9.8e-3/(50**3)/3.0)*x**3 : 1.0/0.0) w l ls 3 title "" ,\
"rk3.dat" using 1:2 ls 1 w p title "Forward Euler",\
"rk3.dat" using 1:3 ls 2 w p title "RK3"

unset output
