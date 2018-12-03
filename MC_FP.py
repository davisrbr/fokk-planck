#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:41:18 2018

@author: ericyelton
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#Explicit file solved using MC integrator 

k = 1
T = 1
hbar = 1.0
m    = 1.0
omega= 1.0

def explicit(W0,D,U,fBND,dx,dt,it,x0,xf):
    J = len(W0)
    W = np.zeros((J+2,it+1))
    
    # I'm sure I could make this two lines, but I didn't want to think
    W[1:J+1,0] = W0
    W[1:J+1,1] = W0
    W[:,0] = fBND(W[:,0])
    W[:,1] = fBND(W[:,1])
    
    # Setting force equal to slope between x+dx/50 and x-dx/50
    #F = np.zeros(J+2)
    #x = np.arange(x0-dx,xf+dx,dx)
    #F = -(U(x+dx)-U(x-dx))/(2*dx)
    
#    for t in range(1,it-1):
#        W[:,t] = fBND(W[:,t])
#        W[1:J+1,t+1] = W[1:J+1,t-1] + 2*dt/(k*T)*(-(F[2:J+2]-F[0:J])/(2*dx)*W[1:J+1,t]-(W[2:J+2,t]-W[0:J,t])/(2*dx)*F[1:J+1]) + 2*dt*D*(W[2:J+2,t]-2*W[1:J+1,t]+W[0:J,t])/(dx**2)
        
    for t in range(0,it):
        W[:,t] = fBND(W[:,t])
        #W[1:J+1,t+1] = W[1:J+1,t] + dt/(k*T)*(-(F[2:J+2]-F[0:J])/(2*dx)*W[1:J+1,t]-(W[2:J+2,t]-W[0:J,t])/(2*dx)*F[1:J+1]) + dt*D*(W[2:J+2,t]-2*W[1:J+1,t]+W[0:J,t])/(dx**2)
        W[1:J+1,t] = integrate_path(x0,xf,0,100,dx,32,4000,5000)[0][:,-1]
    return W

def hamiltonian(x0,x1,eps):
    T = 0.5*((x1-x0)/eps)**2 
    V = Ugiven(x0)
    return T+V

def integrate_path(xa,xb,ta,tb,delta,N,ncut,T):
    # from here ????
    print('N: ', N)
    print('T: ',T)
    X = np.zeros((N,T))
    E = np.zeros((N,T))
    #X[1:N-1,0] = np.random.rand(N-2)
    epsilon = (tb-ta)/N
    E[0,0] = hamiltonian(X[0,-2],xa,epsilon)
    E[0,-1] = E[0,0]
    
    for i in range(1,N):
        E[0,i] = hamiltonian(X[0,i-1],X[0,i],epsilon )

    #iterator to keep track of T:
    j = 0 
    
    while j < T-1: 
        i = np.random.randint(1,N-1,1)

        xi = X[i,j]
        xi_ = X[i-1,j]
        xi_p = X[i+1,j]
        #print(j)
        #print('xi: ',xi,' xi_: ',xi_,' xi_p: ',xi_p)
        u = np.random.rand(1) 
        xp = xi+delta*(2*u-1)
        
        delE = (hamiltonian(xi_,xp,epsilon)+hamiltonian(xp,xi_p,epsilon)-
                hamiltonian(xi_,xi,epsilon)-hamiltonian(xi,xi_p,epsilon))
        #print(delE)
        
        boltzfac = np.exp(-epsilon*delE/hbar)
        r = np.random.rand(1)
        
        if delE < 0:
            #print('made it into first loop')
            #print('xp:', xp)
            j+=1
            X[:,j] = X[:,j-1]
            X[i,j] = xp
            E[:,j] = E[:,j-1]
            E[i,j] += delE 
            
        elif r <= boltzfac: 
            #print('made it into second loop')
            #print('xp:', xp)
            j +=1
            X[:,j] = X[:,j-1]
            X[i,j] = xp
            E[:,j] = E[:,j-1]
            E[i,j] += delE 

        else:
            j+=1 
            X[:,j] = X[:,j-1]
            E[:,j] = E[:,j-1]
    # to here ????
    print('Shape of the output: ', np.shape(X[:,-1]))
    return X[:,ncut:],E[:,ncut:]

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
    x0 = -0
    xf = 1
    J = 32
    dx = (xf-x0)/J
    D = 1
    U = Ugiven
    fBND = Bdirichlet
    it = 10**2
    dt = 10**-6 # Terrible with dt > 10**-5 and super slow with dt < 10**-6
    x = np.arange(x0,xf,dx)
    sig = 0.1
    avg = 0.25#(x0+xf)/2
    W0 = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-avg)**2/(2*sig**2))
    return W0,D,U,fBND,dx,dt,it,x0,xf
    
def Ugiven(x):
    a0 = 300
    a1 = -0.38
    a2 = 1.37
    a3 = -2
    a4 = 1
    return a0*k*T*(a4*x**4 + a3*x**3 + a2*x**2 + a1*x)

def Uwell(x):
    x0 = 0
    x1 = 1
    left = -10**4
    right = 10**4
    xf = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < x0:
            xf[i] = (x[i]-x0)*left
        elif x[i] > x1:
            xf[i] = (x[i]-x1)*right
    return xf

W0,D,U,fBND,dx,dt,it,x0,xf = gaussianSetup()
#W = explicit(W0,D,U,fBND,dx,dt,it,x0,xf)

N = 100
T = 100000
X,E = integrate_path(x0,xf,0,100,0.1,N,1500,T)

xmin = np.min(X)
xmax = np.max(X)

xmin       = np.min(xmin)
xmax       = np.max(xmax)

nbin       = 100
hist,edges = np.histogram(X[1:N-1,:],nbin,range=(xmin,xmax),normed=False)
xbin       = 0.5*(edges[0:edges.size-1]+edges[1:edges.size])
tothist    = np.sum(hist.astype(float))
hist       = hist.astype(float)/tothist

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
#ax = plt.axes(xlim=(x0,xf),ylim=(-3,5))
#line, = ax.plot([],[],lw=2)
#x_vals = np.linspace(x0,xf)
#ax.plot(x_vals, Ugiven(x_vals)+10, color = 'green')
plt.bar(xbin,hist,width=(xbin[1]-xbin[0]),facecolor='green',align='center')
plt.xlabel('$x$')
plt.ylabel('$h(x)$')
#ax.grid()

#for i in range(5):
#    plt.plot(animate(i)[0],label=str(i))
#anim = animation.FuncAnimation(fig, animate, init_func=init,frames=it,interval=1,blit=True)
#plt.legend()
plt.show()

