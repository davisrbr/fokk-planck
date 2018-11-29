import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

k = 1
T = 1

def explicit(W0,D,U,fBND,dx,dt,it,x0,xf):
    J = len(W0)
    W = np.zeros((J+2,it+1))
    
    # I'm sure I could make this two lines, but I didn't want to think
    W[1:J+1,0] = W0
    W[1:J+1,1] = W0
    W[:,0] = fBND(W[:,0])
    W[:,1] = fBND(W[:,1])
    
    # Setting force equal to slope between x+dx/50 and x-dx/50
    F = np.zeros(J+2)
    x = np.arange(x0-dx,xf+dx,dx)
    F = -(U(x+dx/50)-U(x-dx/50))/(dx/100)
    
    for t in range(1,it-1):
        W[:,t] = fBND(W[:,t])
        W[1:J+1,t+1] = W[1:J+1,t-1] + 2*dt/(k*T)*(-(F[2:J+2]-F[0:J])/(2*dx)*W[1:J+1,t]-(W[2:J+2,t]-W[0:J,t])/(2*dx)*F[1:J+1]) + 2*dt*D*(W[2:J+2,t]-2*W[1:J+1,t]+W[0:J,t])/(dx**2)
    
    return W

# input is a J+2 length array
# outputs the same array with W[0] and W[-1] set to boundaries
def Bdirichlet(Wj):
    Wj[0] = -Wj[1]
    Wj[-1] = -Wj[-2]
    return Wj
def Bperiodic(Wj):
    Wj[0] = Wj[-2]
    Wj[-1] = Wj[1]
    return Wj
def Bzero(Wj):
    Wj[0] = 0
    Wj[-1] = 0
    return Wj

def gaussianSetup():
    x0 = 0
    xf = 1
    J = 32
    dx = (xf-x0)/J
    D = 1
    U = Ugiven
    fBND = Bdirichlet
    it = 10**5
    dt = 10**-7 # Terrible with dt > 10**-5 and super slow with dt < 10**-6
    x = np.arange(x0,xf,dx)
    sig = 0.1
    avg = (x0+xf)/2
    W0 = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-avg)**2/(2*sig**2))
    return W0,D,U,fBND,dx,dt,it,x0,xf
    
def Ugiven(x):
    a0 = 300
    a1 = -0.38
    a2 = 1.37
    a3 = -2
    a4 = 1
    return a0*k*T*(a4*x**4 + a3*x**3 + a2*x**2 + a1*x)

W0,D,U,fBND,dx,dt,it,x0,xf = gaussianSetup()
W = explicit(W0,D,U,fBND,dx,dt,it,x0,xf)

# Animation Code Below
def init():
    line.set_data([],[])
    return line,

def animate(i):
    x = np.arange(x0,xf,dx)
    y = W[1:-1,i+1]
    line.set_data(x,y)
    return line,

fig = plt.figure()
ax = plt.axes(xlim=(x0,xf),ylim=(0.05,5))
line, = ax.plot([],[],lw=2)

#for i in range(5):
#    plt.plot(animate(i)[0],label=str(i))
anim = animation.FuncAnimation(fig, animate, init_func=init,frames=it,interval=1,blit=True)
plt.legend()
plt.show()
