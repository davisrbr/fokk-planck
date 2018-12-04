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

def hamiltonian(x0,x1,eps,U):
    T = 0.5*((x1-x0)/eps)**2 
    V = U(x0)
    return T+V

def integrate_path(xa,xb,ta,tb,delta,N,ncut,T,U):
    # from here ????
    print('N: ', N)
    print('T: ',T)
    X = np.zeros((N,T))
    E = np.zeros((N,T))
    #X[1:N-1,0] = np.random.rand(N-2)
    epsilon = (tb-ta)/N
    E[0,0] = hamiltonian(X[0,-2],xa,epsilon,U)
    E[0,-1] = E[0,0]
    
    for i in range(1,N):
        E[0,i] = hamiltonian(X[0,i-1],X[0,i],epsilon,U )

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
        
        delE = (hamiltonian(xi_,xp,epsilon,U)+hamiltonian(xp,xi_p,epsilon,U)-
                hamiltonian(xi_,xi,epsilon,U)-hamiltonian(xi,xi_p,epsilon,U))
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
    U = Ugrav ####******####
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

def Ugrav(x):
    g = 3
    a1 = 412/3
    d1 = -100
    e1 = 927/64
    a2 = 1552/37
    d2 = -2508/37
    e2 = 23571/592
    xf = 0
    if x < 0.25:
        xf = a1*(x**4+x**3+x**2)+d1*x+e1
    elif x > 0.75:
        xf = a2*x**4+d2*x+e2
    else:
        xf = g*x
    return xf

def UgravPlot(x):
    g = 3
    a1 = 412/3
    d1 = -100
    e1 = 927/64
    a2 = 1552/37
    d2 = -2508/37
    e2 = 23571/592
    xf = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] < 0.25:
            xf[i] = a1*(x[i]**4+x[i]**3+x[i]**2)+d1*x[i]+e1
        elif x[i] > 0.75:
            xf[i] = a2*x[i]**4+d2*x[i]+e2
        else:
            xf[i] = g*x[i]
    return xf

def Uconst(x):
    return np.ones(len(x))

def Ubox(x0):
    return  1.5*1e0*((1.0+np.tanh((x0-0.9)/1e-3))+(1.0-np.tanh((x0-0.1)/1e-3))) 

def Umidpeak(x):
    A = 384
    C = -92
    E = 4
    return A*(x-0.5)**4+C*(x-0.5)**2+E

W0,D,U,fBND,dx,dt,it,x0,xf = gaussianSetup()

N = 100
T = 100000
X,E = integrate_path(x0,xf,0,100,0.6,N,1000,T,U)

#==================================================
#IMPORTANT PARAMETERS (use these above)
#-Note that you have to specify the potential in gaussianSetup()
#==================================================
#for the given well N = 100 and T = 10000 with Ncut = 1000 worked well when dx = 0.33
#for the midpeak well N = 100 and T = 100000 with Ncut  = 1000 worked well when dx = 0.6
#for the grav well N = 100 and T = 100000 with Ncut = 1000 worked well when dx 0.6
#for the box well N = 100 and T = 10000 with Ncut = 1000 worked well when dx 0.33
#---------------------------------------------------

xmin = np.min(X)
xmax = np.max(X)

xmin       = np.min(xmin)
xmax       = np.max(xmax)

nbin       = 100
hist,edges = np.histogram(X[1:N-1,:],nbin,range=(xmin,xmax),normed=False)
xbin       = 0.5*(edges[0:edges.size-1]+edges[1:edges.size])
tothist    = np.sum(hist.astype(float))
hist       = hist.astype(float)/tothist

plt.subplot(211)
plt.bar(xbin,hist,width=(xbin[1]-xbin[0]),facecolor='green',align='center', label='Distribution')
plt.xlim(0,1)

plt.ylabel('$h(x)$')
plt.grid()
plt.legend()

plt.subplot(212)
x = np.arange(0,1,1e-2)
if U == Ugrav: 
    plt.plot(x,UgravPlot(x), color = 'r', label='Potential') 
else: plt.plot(x,U(x), color = 'r', label='Potential') 
plt.xlabel('$x$')
plt.ylabel('$U(x)$')
plt.legend()
plt.grid()

plt.savefig('FP_MC_'+U.__name__+'_'+'.png',format = 'png', dpi = 350)

plt.show()



