import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

"""CONFIG"""
k = 1
T = 1
NUM_J_STEPS = 128
ITERATIONS = 10**5
DELTA_T = 10**-5
GAUSSIAN_SIGMA = 0.1
GAUSSIAN_AVG = 0.5
SKIP = 100 #how many values to skip from W
U_FUNC = "Ugiven" #string of function to use
"""U_FUNC Options: Ugiven, Ugrav, Uconst, Ubox, Umidpeak"""
METHOD = "semi_implicit" #string of function to use, defaults to explicit
"""METHOD Options: explicit, semi_implicit, implicit, implicitL, implicitP, chang_cooper"""
SAVE_VIDEO = False
WITH_ERROR = False
FILE_NAME = "TEST_semi_Ugiven.mp4"

"""METHODS"""
def explicit(W0,D,U,fBND,dx,dt,it,x0,xf):
    J = len(W0)
    W = np.zeros((J+2,it+1))
    
    W[1:J+1,0] = W0
    W[:,0] = fBND(W[:,0])
    
    # Setting force equal to slope between x+dx/50 and x-dx/50
    F = np.zeros(J+2)
    x = np.arange(x0-dx,xf+dx,dx)
    F = -(U(x+dx/100)-U(x-dx/100))/(dx/50)
            
    for t in range(0,it):
        W[:,t] = fBND(W[:,t])
        W[1:J+1,t+1] = W[1:J+1,t] + dt/(k*T)*(-(F[2:J+2]-F[0:J])/(2*dx)*W[1:J+1,t]-(W[2:J+2,t]-W[0:J,t])/(2*dx)*F[1:J+1]) + dt*D*(W[2:J+2,t]-2*W[1:J+1,t]+W[0:J,t])/(dx**2)
    return W[:,::SKIP]

def semi_implicit(W0,D,U,fBND,dx,dt,it,x0,xf):
    J = len(W0)
    W = np.zeros((J+2,it+1))
    
    #boundary conditions
    W[1:J+1,0] = W0
    W[:,0] = fBND(W[:,0])

    F = np.zeros(J+2)
    x = (x0-dx + (np.arange(J+2) + 0.5)*dx) #use cell-centered
    F = -(U(x+dx/100)-U(x-dx/100))/(dx/50)
    
    alpha = D*dt/dx**2
    
    A = np.zeros((J,J))
    for i in range(J):
        A[i,i] = 1+2*alpha
        if i!=0:
            A[i-1,i] = -alpha
        if i!=J-1:
            A[i+1,i] = -alpha
    
    temp = np.zeros(J+2)
    B = np.zeros(J)
    for t in range(0,it):
        W[:,t] = fBND(W[:,t])
        temp[1:J+1] = W[1:J+1,t] + (dt/k/T) * (-(F[2:J+2]-F[0:J])*W[1:J+1,t]/(2*dx)-(W[2:J+2,t]-W[0:J,t])*F[1:J+1]/(2*dx))
        temp[:] = fBND(temp)
        B[:] = temp[1:J+1]
        B[0] = B[0]+alpha*temp[0]
        B[-1]= B[-1]+alpha*temp[-1]
        W[1:J+1,t+1] = np.linalg.solve(A,B)
    return W[:, ::SKIP]

"""METHODS BELOW HERE DO NOT FULLY WORK"""
# Solver for a tridiagonal matrix.
# a,b,c are the lower, center, and upper diagonals,
# r is the RHS vector.
def tridiag(a,b,c,r):
    n    = b.size
    gam  = np.zeros(n)
    u    = np.zeros(n)
    bet  = b[0]
    u[0] = r[0]/bet
    for j in range(1,n):
        gam[j] = c[j-1]/bet
        bet    = b[j]-a[j]*gam[j]
        if (bet == 0.0):
            print('[tridiag]: matrix not invertible.')
            exit()
        u[j]   = (r[j]-a[j]*u[j-1])/bet
    for j in range(n-2,-1,-1):
        u[j] = u[j]-gam[j+1]*u[j+1]
    return u

def implicitL(W0,D,U,fBND,dx,dt,it,x0,xf):
    '''Implicit solver: solves fokker-planck implicitly given formulation in Fokker-Planck project paper.
        W0 initial conditions
        D diffusion term
        U potential
        fBND boundary variables
        dx spatial step
        dt time step
        it iterations
        x0 start x
        xf end x'''

    J = len(W0)
    W = np.zeros((J,it))
    alpha = (D*dt)/dx**2  #alpha is dt/(kT*dx)

    # Setting force equal to slope between x+dx/50 and x-dx/50
    F = np.zeros(J+2)
    # x = np.arange(x0-dx,xf+dx,dx) #face-centered
    x = (x0-dx + (np.arange(J+2) + 0.5)*dx) #use cell-centered
    F = -(U(x+dx/100)-U(x-dx/100))/(dx/50)

    W = np.zeros((J+2,it))
    xb = np.zeros(J+2)
    xb = x
    xb[0]    = 2.0*xb[1]-xb[2]
    xb[J+1]  = 2.0*xb[J]-xb[J-1]
    #manually set initial conditions
    sig = 0.1
    avg = (x0+xf)/2
    W[:,0]   = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-(xb-avg)**2/(2*sig**2))

    diaglo = dt*(-1/(2*k*T*dx)*F[1:-1] - D/dx**2) #lower diagonal
    diag = dt*(1/dt + (-F[:-2]+F[2:])/(2*k*T*dx) + (2*D)/dx**2) #main diagonal
    diaghi = dt*(1/(2*k*T*dx)*F[1:-1] - D/dx**2) #upper diagonal
    bvec = np.zeros(J)
    #iterator loop
    for t in range(it-1):
        bvec[:] = W[1:J+1, t]
        W[:,t] = fBND(0, W[:,t])
        W[:,t] = fBND(1, W[:,t])
        W[1:J+1,t+1] = tridiag(diaglo, diag, diaghi, bvec)

    return W[1:J+1,:]

def implicit(W0,D,U,fBND,dx,dt,it,x0,xf):
    '''Implicit solver: solves fokker-planck implicitly given formulation in Fokker-Planck project paper.
        W0 initial conditions
        D diffusion term
        U potential
        fBND boundary variables
        dx spatial step
        dt time step
        it iterations
        x0 start x
        xf end x'''

    J = 128 #len(W0)
    y = np.zeros((J+2,it))
    alpha = (D*dt)/dx**2  #alpha is dt/(kT*dx)
    x = (x0-dx + (np.arange(J) + 0.5)*dx) #use cell-centered
    # initial condition
    xb       = np.zeros(J+2)
    xb[1:J+1]= x
    xb[0]    = 2.0*xb[1]-xb[2]
    xb[J+1]  = 2.0*xb[J]-xb[J-1]
    y[:,0]   = W0
    F = -(U(xb+dx/100)-U(xb-dx/100))/(dx/50)
    # set vectors for tridiag
    diaglo = dt*(-1/(2*k*T*dx)*F[1:-1] - D/dx**2) #lower diagonal
    diag = dt*(1/dt + (-F[:-2]+F[2:])/(2*k*T*dx) + (2*D)/dx**2) #main diagonal
    diaghi = dt*(1/(2*k*T*dx)*F[1:-1] - D/dx**2) #upper diagonal
    bvec = np.zeros(J)
    # integrator loop
    for n in range(it-1):
        bvec[:]      = y[1:J+1,n]
        y[0,n]       = fBND(0,y[:,n])
        y[J+1,n]     = fBND(1,y[:,n])
        y[1:J+1,n+1] = tridiag(diaglo,diag,diaghi,bvec)
    # to here ??????
    return y[1:J+1,:]
def chang_cooper(W0,D,U,fBND,dx,dt,it,x0,xf):
    '''Chang-cooper solver: solves fokker-planck implicitly given formulation in Chang and Cooper (1970).
        W0 initial conditions
        D diffusion term
        U potential
        fBND boundary variables
        dx spatial step
        dt time step
        it iterations
        x0 start x
        xf end x'''

    J = len(W0)
    W = np.zeros((J+2,it+1))
    alpha = (D*dt)/dx**2  #alpha is dt/(kT*dx)

    #boundary conditions
    W[1:J+1,0] = W0
    W[:,0] = fBND(W[:,0])

    # Setting force equal to slope between x+dx/50 and x-dx/50
    F = np.zeros(J+2)
    x = np.arange(x0-dx,xf+dx,dx)
    F = -(U(x+dx/100)-U(x-dx/100))/(dx/50)

    #set up chang cooper 'adaptive stepping'
    w = -F*dx/D
    delta = 1/w - 1/(np.exp(w) - 1)

    #set up tridiagonals w/ chang cooper stepping
    c = dt*(-1/(2*k*T*dx)*F[1:-1]*(1-delta[1:-1]) - D/dx**2) #lower diagonal
    b = dt*(1/dt + (-F[:-2]*delta[1:-1]+F[2:]*(1-delta[2:]))/(2*k*T*dx) + (2*D)/dx**2) #main diagonal
    a = dt*(1/(2*k*T*dx)*F[1:-1]*delta[2:] - D/dx**2) #upper diagonal

    ##add no flux conditions NOTE: check correctness
    #c[0 ] = (-1/(2*k*T*dx)*F[0] + D/dx**2)
    #c[-2] = (-1/(2*k*T*dx)*F[-1] + D/dx**2)
    #b[0 ] = ((-F[1]+F[0])/(2*k*T*dx) - (2*D)/dx**2)
    #b[-1] = ((-F[-2]+F[-1])/(2*k*T*dx) - (2*D)/dx**2)
    #a[0 ] = (-1/(2*k*T*dx)*F[0] + D/dx**2)
    #a[-2] = (-1/(2*k*T*dx)*F[-1] + D/dx**2)

    diagonals = [b, a, c] #put in list in order to create sparse matrix
    A = diags(diagonals, [0, 1, -1]).toarray() #construct sparse tridiagonal matrix

    for t in range(1,it):
        r = np.linalg.solve(A, W[1:-1,t-1])
        W[1:-1,t] = r #set solution

    return W
def implicitP(W0,D,U,fBND,dx,dt,it,x0,xf):
    '''Implicit solver: solves fokker-planck implicitly given formulation in Park and Petrosian.
        W0 initial conditions
        D diffusion term
        U potential
        fBND boundary variables
        dx spatial step
        dt time step
        it iterations
        x0 start x
        xf end x'''

    J = len(W0)
    W = np.zeros((J+2,it+1))
    alpha = (D*dt)/dx**2  #alpha is dt/(kT*dx)

    #boundary conditions
    W[1:J+1,0] = W0
    W[:,0] = fBND(W[:,0])

    # Setting force equal to slope between x+dx/50 and x-dx/50
    F = np.zeros(J+2)
    x = np.arange(x0-dx,xf+dx,dx)
    F = -(U(x+dx/100)-U(x-dx/100))/(dx/50)

    w = -F[1:-1]/D * dx
    #set up tridiagonals using Park and Petrosian
    c = alpha*(1-w/2) #lower diagonal
    b = 1 + alpha*((1+w/2) + (1-w/2)) #main diagonal
    a = alpha*(1+w/2) #upper diagonal

    #add no flux conditions
    # c[0 ] = (-1/(2*k*T*dx)*F[0] + D/dx**2)
    # c[-2] = (-1/(2*k*T*dx)*F[-1] + D/dx**2)
    # b[0 ] = ((-F[1]+F[0])/(2*k*T*dx) - (2*D)/dx**2)
    # b[-1] = ((-F[-2]+F[-1])/(2*k*T*dx) - (2*D)/dx**2)
    # a[0 ] = (-1/(2*k*T*dx)*F[0] + D/dx**2)
    # a[-2] = (-1/(2*k*T*dx)*F[-1] + D/dx**2)

    diagonals = [b, a, c] #put in list in order to create sparse matrix
    A = diags(diagonals, [0, 1, -1]).toarray() #construct sparse tridiagonal matrix

    for t in range(1,it):
        W[:,t] = fBND(W[:,t-1])
        r = np.linalg.solve(A, W[1:-1,t-1])
        W[1:-1,t] = r #set solution
    return W

"""BOUNDARY CONDITIONS"""
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
def Bslope(Wj):
    Wj[0] = Wj[1] - (Wj[2]-Wj[1])
    Wj[-1] = Wj[-2] + (Wj[-3] - Wj[-2])

def setup():
    x0 = -0
    xf = 1
    J = NUM_J_STEPS
    dx = (xf-x0)/J
    D = 1

    if U_FUNC == "Ugiven":
        U = Ugiven
    elif U_FUNC == "Ugrav":
        U = Ugrav
    elif U_FUNC == "Uconst":
        U = Uconst
    elif U_FUNC == "Ubox":
        U = Ubox
    elif U_FUNC == "Umidpeak":
        U = Umidpeak
    else:
        U_FUNC == Ugiven

    fBND = Bzero
    it = ITERATIONS
    dt = DELTA_T
    x = np.arange(x0,xf,dx)
    sig = GAUSSIAN_SIGMA
    avg = GAUSSIAN_AVG
    W0 = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-avg)**2/(2*sig**2))
#    W0 = np.ones(J)
    return W0,D,U,fBND,dx,dt,it,x0,xf
    
def Ugiven(x):
    a0 = 300
    a1 = -0.38
    a2 = 1.37
    a3 = -2
    a4 = 1
    return a0*k*T*(a4*x**4 + a3*x**3 + a2*x**2 + a1*x)

def Ugrav(x):
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

"""METHOD"""
W0,D,U,fBND,dx,dt,it,x0,xf = setup()

if METHOD == "explicit":
    W = explicit(W0,D,U,fBND,dx,dt,it,x0,xf)
elif METHOD == "semi_implicit":
    W = semi_implicit(W0,D,U,fBND,dx,dt,it,x0,xf)
elif METHOD == "implicit":
    W = implicit(W0,D,U,fBND,dx,dt,it,x0,xf)
elif METHOD == "implicitL":
    W = implicitL(W0,D,U,fBND,dx,dt,it,x0,xf)
elif METHOD == "chang_cooper":
    W = chang_cooper(W0,D,U,fBND,dx,dt,it,x0,xf)
elif METHOD == "implicitP":
    W = implicitP(W0,D,U,fBND,dx,dt,it,x0,xf)
else:
    W = explicit(W0,D,U,fBND,dx,dt,it,x0,xf)

"""AREA CALCS"""
def areaCalc(W):
    area = np.zeros(len(W[0]))
    for i in range(len(W[0])):
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
print("Error sum:", np.sum(Err[-1]))

"""ANIMATION"""
def init():
    line.set_data([],[])
    return line,

def animate(i):
    x = np.arange(x0,xf,dx)
    y = W[1:-1,i]
    line.set_data(x,y)
    line.set_label("Numerical")
    plt.legend()
    return line,

def init_with_error():
    line.set_data([],[])
    line2.set_data([],[])
    return [line,line2]

def animate_with_error(i):
    x = np.arange(x0,xf,dx)
    y = W[:,i+1]
    y2 = Err[:,i+1]
    line.set_data(x,y)
    line.set_label("Numerical")
    line2.set_data(x,y2)
    line2.set_label("Error")
    line2.set_color("r")
    # legend.remove()
    legend = plt.legend()

    return [line,line2]# + [legend]



"""PLOTTING"""
if not SAVE_VIDEO:
    x_vals = np.linspace(x0,xf,512)
    sol = np.exp(-U(x_vals)/D)
    solA = area1D(sol,(xf-x0)/512)
    sol = sol/solA
    print("solA",solA)


    if not WITH_ERROR:
        fig = plt.figure()
        ax = plt.axes(xlim=(x0,xf),ylim=(-0.5,5))
        line, = ax.plot([],[],lw=2)
        ax.plot(x_vals, U(x_vals)-min(U(x_vals)), color = 'green',label='Potential')
        ax.plot(x_vals, sol,'k',label='Analytic')

        ax.grid()

        anim = animation.FuncAnimation(fig, animate, init_func=init,frames=len(W[0]),interval=1,blit=True)
    else:
        fig = plt.figure()
        ax = plt.axes(xlim=(x0,xf),ylim=(-5,5))
        line, = ax.plot([],[],lw=2)
        line2, = ax.plot([],[],lw=2)
        ax.plot(x_vals, U(x_vals)-min(U(x_vals)), color = 'green',label='Potential')
        ax.plot(x_vals, sol,'k',label='Analytic')

        ax.grid()
        legend = plt.legend()

        anim = animation.FuncAnimation(fig, animate_with_error, init_func=init_with_error,frames=it,interval=1,blit=True)

    plt.legend()

    plt.show()
else:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Eric_$$$_Yelton'))

    x_vals = np.linspace(x0,xf,512)
    sol = np.exp(-U(x_vals)/D)
    solA = area1D(sol,(xf-x0)/512)
    sol = sol/solA

    fig = plt.figure()
    ax = plt.axes(xlim=(x0,xf),ylim=(-0.5,5))
    line, = ax.plot([],[],lw=2)
    ax.plot(x_vals, U(x_vals)-min(U(x_vals)), color = 'green',label='Potential')
    ax.plot(x_vals, sol,'k',label='Analytic')
    ax.grid()

    if not WITH_ERROR:
        anim = animation.FuncAnimation(fig, animate, init_func=init,frames=len(W[0]),interval=1,blit=True)
    else:
        anim = animation.FuncAnimation(fig, animate_with_error, init_func=init_with_error,frames=len(W[0]),interval=1,blit=True)
    
    anim.save(FILE_NAME, writer=writer,dpi = 350)
    plt.legend(loc = 0)

    plt.show()

"""IDK"""
def peaks(W):
    peaks=[]
    for n in range(len(W[0])-2):
        p = []
        for j in range(len(W)-1):
            if W[j,n] > W[j-1,n] and W[j,n] > W[j+1,n] and W[j,n] > 1:
                p.append(W[j,n])
        peaks.append(p)
    return peaks