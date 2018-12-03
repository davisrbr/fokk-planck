#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 18:53:03 2018

@author: ericyelton
"""

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
#ax.plot(mid,np.ones(it),'.r',label='midpoint')
ax.grid()
#for i in range(5):
#    plt.plot(animate(i)[0],label=str(i))
anim = animation.FuncAnimation(fig, animate, init_func=init,frames=it,interval=1,blit=True)
anim.save('Given_Well.mp4', writer=writer,dpi = 350)
plt.legend(loc = 0)

plt.show()