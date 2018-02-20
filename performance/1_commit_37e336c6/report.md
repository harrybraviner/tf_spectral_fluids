# Forward Euler with no CFL condition

The state of the code as of this commit has only forward Euler timestepping.
It has no CFL condition (so timestep is constant) and also lacks aliasing masking.
I did some runs just to see how the time-per-step varied between the CPU and GPU.

![Time-per-step plot](./time_per_step.png)
