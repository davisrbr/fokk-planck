import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.sparse import diags
k = 1.
T = 1.

def implicitL(W0,D,U,fBND,dx,dt,it,x0,xf):
    '''Implicit solver
            W0 initial conditions
            D diffusion term
            U potential
            fBND boundary variables
            '''
    J = len(W0)
    W = np.zeros((J+2,it+1))
    alpha = (D*dt)/dx**2  #alpha is dt/(kT*dx)

    # I'm sure I could make this two lines, but I didn't want to think
    W[1:J+1,0] = W0
    W[1:J+1,1] = W0
    W[:,0] = fBND(W[:,0])
    W[:,1] = fBND(W[:,1])

    # Setting force equal to slope between x+dx/50 and x-dx/50
    F = np.zeros(J+2)
    x = np.arange(x0-dx,xf+dx,dx)
    F = -(U(x+dx/100)-U(x-dx/100))/(dx/50)

    #initialize zeros for tridiagonal arrays
    c = np.zeros(J)
    b = np.zeros(J)
    a = np.zeros(J)

    #set up tridiagonals
    c = dt*(-1/(2*k*T*dx)*F[1:-1] - D/dx**2)
    b = dt*(1/dt + (-F[:-2]+F[2:])/(2*k*T*dx) + (2*D)/dx**2)
    a = dt*(1/(2*k*T*dx)*F[1:-1] - D/dx**2)

    #add no flux conditions
    c[0 ] = (-1/(2*k*T*dx)*F[0] + D/dx**2)
    c[-1] = (-1/(2*k*T*dx)*F[-1] + D/dx**2)
    b[0 ] = ((-F[1]+F[0])/(2*k*T*dx) - (2*D)/dx**2)
    b[-1] = ((-F[-2]+F[-1])/(2*k*T*dx) - (2*D)/dx**2)
    a[0 ] = (-1/(2*k*T*dx)*F[0] + D/dx**2)
    a[-1] = (-1/(2*k*T*dx)*F[-1] + D/dx**2)


    diagonals = [b, a, c] #put in list in order to create sparse matrix
    A = diags(diagonals, [0, 1, -1]).toarray() #construct sparse tridiagonal matrix

    for t in range(1,it):
        r = np.linalg.solve(A, W[1:-1,t-1])
        #r = fBND(r) #enforce b.c.s
        #input()
        #print(r)
        W[1:-1,t] = r #set solution

    return W

def chang_cooper(W0,D,U,fBND,dx,dt,it,x0,xf):
    J = len(W0)
    W = np.zeros((J+2,it+1))
    alpha = (D*dt)/dx**2  #alpha is dt/(kT*dx)


    # I'm sure I could make this two lines, but I didn't want to think
    W[1:J+1,0] = W0
    W[1:J+1,1] = W0
    W[:,0] = fBND(W[:,0])
    W[:,1] = fBND(W[:,1])

    # Setting force equal to slope between x+dx/50 and x-dx/50
    F = np.zeros(J+2)
    x = np.arange(x0-dx,xf+dx,dx)
    F = -(U(x+dx/50)-U(x-dx/50))/(dx/100)

    a = np.zeros(J)
    b = np.zeros(J)
    c = np.zeros(J)

    w = -F[1:-1]/D * dx

    #define new variables necessary for flux conservation
    delta_m = np.power(w,-1) - np.power(np.exp(w)-1, -1)
    W_m = (w/2)/(np.sinh(w/2)) #may need to be rewritten for computational use (overflow may occur)
    W_plus = W_m * np.exp(w/2)
    W_minus = W_m * np.exp(-w/2)


    for t in range(1,it-1):
        a = alpha * W_plus
        b = 1 + alpha * (W_plus+W_minus) #may be ignoring important aspect of m+/- 1/2
        c = alpha * W_minus

        #construct sparse tridiagonal matrix
        diagonals = [b, a, c]
        A = diags(diagonals, [0, 1, -1]).toarray()
        r = np.linalg.solve(A, W[1:-1,t-1])

        #set solution
        W[1:-1,t] = r


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
    J = 128
    dx = (xf-x0)/J
    D = 1
    U = Ugiven
    fBND = Bdirichlet
    it = 10**4
    dt = 10**-5 # Terrible with dt > 10**-5 and super slow with dt < 10**-6
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

W = implicitL(W0,D,U,fBND,dx,dt,it,x0,xf)
# W = chang_cooper(W0,D,U,fBND,dx,dt,it,x0,xf)

def areaCalc(W):
    area = np.zeros(it)
    for i in range(it):
        area[i] = dx*np.sum(W[:,i])
    return area

def area1D(W,dx):
    area = dx*np.sum(W)
    return area

def midCalc(W,x):
    mid = np.zeros(it)
    for i in range(it):
        mid[i] = np.sum(dx*x*W[1:-1,i])
    return mid

area = areaCalc(W)
print("Initial Area:",area[0])
print("Final Area:",area[-1])

def error(W):
    # W# J x T
    x_vals = np.linspace(x0,xf,len(W))
    sol = np.exp(-U(x_vals)/D)
    solA = area1D(sol,(xf-x0)/len(W))
    sol = sol/solA

    return (W.T - sol.T).T
Err = error(W)
print("Error:", Err)


#x = np.arange(x0,xf,dx)
#mid = midCalc(W,x)


#def animateMid(i):
#    x = mid[i]
#    y = np.ones(it)
#    line.set_data(x,y)
#    return line,

x_vals = np.linspace(x0,xf,512)
sol = np.exp(-U(x_vals)/D)
solA = area1D(sol,(xf-x0)/512)
sol = sol/solA

fig = plt.figure()
ax = plt.axes(xlim=(x0,xf),ylim=(-5,5))
line, = ax.plot([],[],lw=2)
line2, = ax.plot([],[],lw=2)
ax.plot(x_vals, U(x_vals)-min(U(x_vals)), color = 'green',label='Potential')
ax.plot(x_vals, sol,'k',label='Analytic')

#ax.plot(mid,np.ones(it),'.r',label='midpoint')
ax.grid()
legend = plt.legend()

#for i in range(5):
#    plt.plot(animate(i)[0],label=str(i))
# Animation Code Below
def init():
    line.set_data([],[])
    line2.set_data([],[])
    return [line,line2]

def animate(i):
    x = np.arange(x0,xf,dx)
    y = W[1:-1,i+1]
    y2 = Err[1:-1,i+1]
    line.set_data(x,y)
    line.set_label("Numerical")
    line2.set_data(x,y2)
    line2.set_label("Error")
    line2.set_color("r")
    # legend.remove()
    legend = plt.legend()

    return [line,line2]# + [legend]

anim = animation.FuncAnimation(fig, animate, init_func=init,frames=it,interval=1,blit=True)

plt.show()

