from __future__ import division

import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.patches as pat
from matplotlib import rc
## Parameters
k      = (2*np.pi)       # Lengths will be in units of wavelength
p0     = 1        # Pressure amplitude of each source
xRange = 2  
yRange = 2
num    = 1000      # Number of points in each direction


## Coordinate Arrays
xL = np.linspace(-xRange/2,xRange/2,num=num)
yL = np.linspace(-yRange/2,yRange/2,num=num)
xG, yG = np.meshgrid(xL,yL) 

## Plane Standing Wave (two sources)
p2 = (p0/2) * (np.exp(1j*k*xG) + np.exp(-1j*k*xG) )

## Three Standing Waves? (six sources)
p6 = (p0/6)*( np.exp(-1j*k*xG) +np.exp(1j*k*xG) + np.exp(1j*k *( (1/2)*xG -(np.sqrt(3)/2)*yG) )+ np.exp(-1j*k *( (1/2)*xG -(np.sqrt(3)/2)*yG) ) + np.exp(1j*k*( (1/2)*xG + (np.sqrt(3)/2)*yG ) )+np.exp(-1j*k*( (1/2)*xG + (np.sqrt(3)/2)*yG ) ) )


## Three Momentum-Cancelling Plane Waves (three sources)
def p3(xG,yG):
  p3result = (p0/3) * ( np.exp(-1j*k*xG) + np.exp(1j*k *( (1/2)*xG -(np.sqrt(3)/2)*yG) ) + np.exp(1j*k*( (1/2)*xG + (np.sqrt(3)/2)*yG ) ) )
  return p3result

phase3 = np.angle( p3(xG,yG) )
phase3Grad = np.gradient(phase3)

reduceFac = int(num/50)
xGreduced = xG[0::reduceFac,0::reduceFac]
yGreduced = yG[0::reduceFac,0::reduceFac]
phase3reduced = np.angle( p3(xGreduced,yGreduced) )
phase3GradreducedUnNorm = np.gradient(phase3reduced)
phase3Gradreduced= 10*phase3GradreducedUnNorm/(np.sqrt( (phase3GradreducedUnNorm[0])**2 + (phase3GradreducedUnNorm[1])**2) )

## INTENSITY PLOTS

plt.rcParams.update({'font.size':16})

ax1=plt.subplot(3,1,1,aspect='equal')
plt.pcolormesh(xG,yG, np.abs(p2)**2, cmap='viridis')
plt.xlim([-xRange/2,xRange/2])
plt.ylim([-yRange/2,yRange/2])
plt.title('Intensity')
plt.ylabel(r'$y/\lambda$ ')
plt.colorbar()

ax2=plt.subplot(3,1,2,aspect='equal')
plt.pcolormesh(xG,yG, np.abs(p6)**2, cmap='viridis')
plt.xlim([-xRange/2,xRange/2])
plt.ylim([-yRange/2,yRange/2])
plt.ylabel(r'$y/\lambda$ ')
plt.colorbar()

ax3=plt.subplot(3,1,3,aspect='equal')
plt.pcolormesh(xG,yG, np.abs( p3(xG,yG) )**2, cmap='viridis')
plt.xlim([-xRange/2,xRange/2])
plt.ylim([-yRange/2,yRange/2])
plt.ylabel(r'$y/\lambda$ ')
plt.colorbar()



#Create circles on each node 
circles = []
rad = 0.1
R = 4*np.pi/(3*np.sqrt(3)*k)
lw = 2


centers=[]
centers.append( np.array((0,R)) ) #Calculated one node, others are just 60 degree rotations from it
theta = np.radians(60)
for i in range(1,6):
	newCenter = (np.cos(theta)*centers[i-1][0]-np.sin(theta)*centers[i-1][1],np.sin(theta)*centers[i-1][0]+np.cos(theta)*centers[i-1][1])
	centers.append( newCenter ) #Do the rotations, put a circle around each point

for i in range(6):
	circles.append( pat.Circle(centers[i],  rad , fill=False, lw=3,ls=':', color='white') ) #Draw those circles


arrows = []
connect = pat.ConnectionStyle("Arc3",rad=.375)
head = pat.ArrowStyle("<-",head_length=4,head_width=4)
for i in range(6):
	arrows.append( pat.FancyArrowPatch( posA=( centers[i]-np.array((rad,0))), posB= ( centers[i] -np.array((0,rad))) , arrowstyle=head, color='white',connectionstyle=connect ,lw=lw) ) 
for i in range(6):
	ax3.add_patch( circles[i] )
	#ax3.add_patch( arrows [i]  )

plt.xlabel(r'$x/\lambda$ ')
f=plt.gcf()
f.set_size_inches(8,32,forward=True)

plt.show()

## PHASE PLOT

ax4=plt.subplot(1,1,1,aspect='equal')
plt.pcolormesh(xG,yG, phase3, cmap='hsv')

	
plt.xlim([-xRange/4,xRange/4])
plt.ylim([-yRange/4,yRange/4])
plt.title('Phase')
plt.ylabel(r'$y/\lambda$ ')
plt.xlabel(r'$x/\lambda$ ')
plt.colorbar()
circles = []
#for i in range(6):
	#circles.append( pat.Circle(centers[i],  rad , fill=False, lw=3,ls=':', color='black') ) #Draw those circles
#for i in range(6):
	#ax4.add_patch( circles[i] )
# STREAMPLOT 
ax5 = plt.subplot(1,1,1,aspect='equal')
streamPoints = np.zeros((num,2))
streamPoints[:,0] =  np.linspace(-xRange/2,xRange/2,num)
streamPoints[:,1] = -np.linspace(-yRange/2,yRange/2,num)

ax4.streamplot(xG,yG, phase3Grad[1] ,  phase3Grad[0] ,color='black', density=3, arrowstyle='fancy' )

#plt.xlim([-xRange/2,xRange/2])
#plt.ylim([-yRange/2,yRange/2])
#plt.title('Phase Gradient')
#plt.ylabel(r'$y/\lambda$ ')
#plt.xlabel(r'$x/\lambda$ ')	

plt.show()

## ACOUSTIC POTENTIAL




