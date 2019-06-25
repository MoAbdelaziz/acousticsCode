from __future__ import division

import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.patches as pat
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc

## Parameters

xRange = 2  
yRange = 2
num    = 1000     # Number of points in each direction


## Coordinate Arrays
xL = np.linspace(-xRange/2,xRange/2,num=num)
yL = np.linspace(-yRange/2,yRange/2,num=num)
xG, yG = np.meshgrid(xL,yL) 

def xC(r,theta): # polar to Cartesian
    return r*np.cos(theta)
def yC(r,theta):
    return r*np.sin(theta)
  
def rC(x,y): # Cartesian to polar
    return np.sqrt(x**2 + y**2)
def thetaC(x,y):
    return np.arctan2(y,x)
  
  
def field(r):
    return r**2
def phase(theta):
    return theta % (2*np.pi)


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')



ax.plot_surface(xG,yG, field(rC(xG,yG)), cmap='hsv', facecolors = cm.hsv( phase(thetaC(xG,yG))/(2*np.pi) ), rstride=10 ,cstride=10) #Draw the phase coloration
ax.plot_surface(xG,yG, field(rC(xG,yG)) , rstride=100 ,cstride=100, cmap=None,alpha=.1) #Draw the gridlines (in a weird way)

## Draw the sphere
u, v = np.mgrid[0:np.pi:50j, 0:2*np.pi:50j]


x0=.25
y0=.25
z0=x0**2+y0**2+0.1
R=.25

x = R * np.sin(u) * np.cos(v)+x0
y = R * np.sin(u) * np.sin(v)+y0
z = R * np.cos(u)+z0

ax.plot_surface(x, y, z, rstride=1, cstride=1, color='k',
                       linewidth=0, antialiased=False,zorder=1)
##

ax.set_axis_off()



#ax.xaxis.set_ticklabels([])

#ax.xaxis.set_ticks([])
#ax.yaxis.set_ticks([])
#ax.zaxis.set_ticks([])
#ax.yaxis.set_ticklabels([])
#ax.zaxis.set_ticklabels([])
ax.set_xlim3d( [-xRange/2, xRange/2] )
ax.set_ylim3d( [-yRange/2, yRange/2] )
ax.set_zlim3d( [0,         field(rC(xRange/2,yRange/2))])
ax.view_init(45,160)
fig.tight_layout()

plt.show()