from __future__ import division

import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.patches as pat
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
from mayavi import mlab

## Parameters

xRange = 10
yRange = 10
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
    return .5*r**2*np.exp(-r/3)
def phase(theta):
    return theta % (2*np.pi)


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')



ax.plot_surface(xG,yG, field(rC(xG,yG)), cmap='hsv', facecolors = cm.hsv( phase(thetaC(xG,yG))/(2*np.pi) ), rstride=10 ,cstride=10) #Draw the phase coloration
ax.plot_surface(xG,yG, field(rC(xG,yG)) , rstride=100 ,cstride=100, cmap=None,alpha=.1) #Draw the gridlines (in a weird way)

## Draw the sphere
u, v = np.mgrid[0:np.pi:50j, 0:2*np.pi:50j]

R=1.1

x0=-R
y0=R
z0=x0**2+y0**2+R/2

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

#plt.show()


## Using Mayavi2/mlab instead of matplotlib/mplot3d ##
mlab.figure(bgcolor=(1,1,1))
res=100 # sparseness of wireframe relative to full surface data. Higher number, more sparse
trap = mlab.mesh(xG,yG, field (rC (xG,yG ) ) ,scalars =  phase(thetaC(xG,yG))/(2*np.pi), colormap = 'hsv')
#trapBacking = mlab.mesh(xG,yG, field (rC (xG,yG ) )-0.01 ,color=(0,0,0) )
#trapFrame = mlab.mesh(xG[0::res,0::res],yG[0::res,0::res], field (rC (xG,yG )[0::res,0::res] ) ,color = (0,0,0) , representation = 'wireframe', mode='axes',opacity=.1)
#trapFrame = mlab.surf(xG[0::res,0::res],yG[0::res,0::res], field (rC (xG,yG )[0::res,0::res] ))

#radiusLine = mlab.plot3d([x0+R,x0+R],[y0+R,y0+R],[z0,z0+R],color=(0,0,0) ) # Draw radius of the sphere
sphere = mlab.mesh(x,y,z, color=(1,1,1) )
mlab.view(distance=20)
drawNum = int(num/res)
for i in range(0,drawNum): #Just draw num/res lines for a custom square gridding...
	xPoints1 = np.ones(drawNum)*i*(xRange/drawNum)-xRange/2
	yPoints1 = np.linspace(-yRange/2,yRange/2,drawNum)
	zPoints1 = field(rC(xPoints1,yPoints1))
	mlab.plot3d(xPoints1 , yPoints1  , zPoints1,color=(0,0,0),opacity=0.1)
	
	xPoints2 = np.linspace(-xRange/2,xRange/2,drawNum)
	yPoints2 = np.ones(drawNum)*i*(yRange/drawNum)-yRange/2
	zPoints2 = field(rC(xPoints2,yPoints2))
	mlab.plot3d(xPoints2 , yPoints2  , zPoints2,color=(0,0,0),opacity=0.1)

#mlab.axes(extent=[-xRange/2,xRange/2,-yRange/2,yRange/2,0,field(rC(xRange/2,yRange/2))])


mlab.savefig(filename = 'harmonicVortex.png')
mlab.view(distance=20)

mlab.show()
