set terminal png size 640, 480
set output "vs_snoopy.png"

set linestyle 1 ps 1 pt 7
set linestyle 2 ps 1 pt 5
set style line 3 lw 1 lt rgb "gray"

set xlabel "Collocation points in each direction"
set ylabel "Wall-clock time per timestep"
set logscale x
set logscale y

set xtics (8, 16, 32, 64, 128, 256)

plot (x > 64 ? 8.29e-7*x**3 : 1.0/0.0) w l ls 3 title "N^3" ,\
"../snoopy_timings/snoopy_timings_laptop.dat" using 1:3 ls 1 w p title "Snoopy, 4 CPUs, 16GB memory",\
"../snoopy_timings/tf_gpu_desktop.dat" using 1:2 ls 2 w p title "tf, 1050Ti 4GB, commit 3707d467"

unset output

