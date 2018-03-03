#! /usr/bin/python

# This code produces the mapping graph for the u_n+1 ( u_n ) mapping.
# This may be used to infer stability.

import numpy as np
import matplotlib.pyplot as plt

nu = 1.0
k2 = (2.0*np.pi)**2
f=1.0
N = 32

u_min = 1e-3
u_max = 1.0

def h(u):
    return 1.5 / (2.0*np.pi*(N//2 - 1) * u)

def u(u):
    return np.exp(-nu*k2*h(u)) * (u + h(u)*f)

u_grid = np.linspace(start=u_min, stop=u_max, num=100)
y_grid = np.vectorize(u)(u_grid)

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(u_grid, y_grid)
ax.plot([u_min, u_max], [u_min, u_max])
fig.savefig("stability.png")
