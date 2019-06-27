## Similar idea as plot in sphereFigure.py, but just do a 2D projection since 
## the 3D has a lot of unnecessary space and looks weird. Color can indicate the
## phase and contour line density (+ arrows?) for density

from __future__ import division

import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.patches as pat
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from mayavi import mlab
import matplotlib.colors as col
import cmocean
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
    return .5*r**2*np.exp(-r/.3)
def phase(theta):
    return theta % (2*np.pi)
fig, ax = plt.subplots()
## Draw the particle
x0 = 0.5
y0 = 0.5
R  = 0.3
circleCenter = (x0,y0)
circle = plt.Circle( circleCenter , R, color='white', ec='black',linewidth=2)

intArrow = plt.arrow(x0-R/np.sqrt(2),y0-R/np.sqrt(2),-0.2,-0.2,lw=5)
intArrowIn = plt.arrow(x0-R/np.sqrt(2),y0-R/np.sqrt(2),-0.2,-0.2,lw=1,color='white')
intText = plt.text(x0-R,y0-1.5*R,r"$\nabla u^2$",fontsize=24,color='white') 

phsArrow = plt.arrow( x0+R/np.sqrt(2), y0 -R/np.sqrt(2), 0.15, -0.15,lw=5)
phsArrowIn = plt.arrow( x0+R/np.sqrt(2), y0 -R/np.sqrt(2), 0.15, -0.15,lw=1,color='white')
phsText = plt.text(x0+0.5*R,y0-1.6*R,r"$\nabla \varphi$",fontsize=24,color='white')

radLine = plt.plot([x0,x0+R/np.sqrt(2)],[y0,y0+R/np.sqrt(2)],color='black',linewidth=2)# Draw its radius
plt.text(0.05+x0+R/(2*np.sqrt(2)),-0.05+y0+R/(2*np.sqrt(2)),r"$a_p$", fontsize=24) # label radius
plt.text(x0-R/2,y0-R/2,r"$\rho_p \, , c_p \,$",fontsize=24)# Label its rho, c
plt.text(-xRange*3/8,-yRange*3/8, r"$\rho_m \, , c_m$",fontsize=24,color='white')# Label medium properties

plt.contour(xG,yG, field( rC(xG,yG) ) ,colors = ['black'], zorder=0 )# Contour lines of field intensity
#plt.contourf(xG,yG, phase( thetaC(xG,yG) ), 360 ,cmap=cmocean.cm.phase, zorder=-1, linewidth=0,vmin=0,vmax=2*np.pi)
plt.pcolormesh(xG,yG, phase( thetaC(xG,yG) ),cmap=cmocean.cm.phase,zorder=-1)
ax.add_artist(circle)
plt.xlim([-xRange/2,xRange/2])
plt.ylim([-yRange/2,yRange/2])
plt.axis('off')
ax.set(aspect='equal')
plt.show()



