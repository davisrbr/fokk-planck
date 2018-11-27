import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy import signal
from scipy import stats

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# Below are the possible 'solutions' to plot
# Sine wave used to test animation
def sine():
    def func(i):
        x = np.linspace(0, 2, 1000)
        y = np.sin(2 * np.pi * (x - 0.01 * i))
        line.set_data(x, y)
        return line,
    xbnds = (0,2)
    ybnds = (-1,1)
    frames = 200
    interval = 20
    return func, xbnds, ybnds, frames, interval

# One point width pulse moving from 0 to 2
def delta():
    def func(i):
        x = np.linspace(0, 2, 1000)
        y = signal.unit_impulse(1000,i)
        line.set_data(x,y)
        return line,
    xbnds = (0,2)
    ybnds = (-0.05,1.2)
    frames = 1000
    interval = 4
    return func, xbnds, ybnds, frames, interval

# One point width pulse moving according to kinematic equations
def grav():
    def func(i):
        x = np.linspace(0, 2, 1000)
        g  = 1
        v0 = -30
        y0 = 500
        pos = int(0.5*g*i**2 + v0*i + y0)
        y = signal.unit_impulse(1000,pos)
        line.set_data(x,y)
        return line,
    xbnds = (0,2)
    ybnds = (-0.05,1.2)
    frames = 200
    interval = 20
    return func, xbnds, ybnds, frames, interval

# Normal distribution that spreads out with time
def normDiff():
    def func(i):
        x = np.linspace(0, 2, 1000)
        mean = 1.
        s0 = 0.001
        y = stats.norm(mean,s0+i/200).pdf(x)
        line.set_data(x,y)
        return line,
    xbnds = (0,2)
    ybnds = (-0.05,1.2)
    frames = 200
    interval = 20
    return func, xbnds, ybnds, frames, interval

# Normal distribution that has a mean moving according to kinematic equations that spreads out over time
def gravDiff():
    def func(i):
        x = np.linspace(0,2,1000)
        g  = 1
        v0 = 0
        y0 = 0
        it = i/100
        mean = 0.5*g*it**2 + v0*it + y0
        s0 = 0.001
        y = stats.norm(mean,s0+i/500).pdf(x)
        line.set_data(x,y)
        return line,
    xbnds = (0,2)
    ybnds = (-0.05,2.0)
    frames = 200
    interval = 20
    return func, xbnds, ybnds, frames, interval

# Analytical solution to diffusion equation with constant diffusion
def diffSol():
    def func(i):
        x = np.linspace(0,2,1000)
        t = i/500
        D = 1
        y = (1/np.sqrt(np.pi*4*D*t))*np.exp(-(x-1)**2/(4*D*t))
        line.set_data(x,y)
        return line,
    xbnds = (0,2)
    ybnds = (-0.05,1.2)
    frames = 200
    interval = 20
    return func, xbnds, ybnds, frames, interval

# Change this line to change which function to plot
animate, xbnds, ybnds, frames, interval = diffSol()

fig = plt.figure()
ax = plt.axes(xlim=xbnds, ylim=ybnds)
line, = ax.plot([], [], lw=2)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=interval, blit=True)

plt.show()