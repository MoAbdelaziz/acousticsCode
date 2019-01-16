from __future__ import division
import numpy as np
import pylab as plt

### Parameters ###
lam = 1.          # wavelength (define as unity)
k   = 2*np.pi/lam # wavenumber
z   = 5	           # Distance from P0 to P1 in wavelengths
# P0 is the diffraction plane and P1 is the imaging plane

### Generate Scalar Field in Plane P0 ###
P0L = 5.6 # Side length of P0 in wavelengths
P0N = 100 # Number of points per side
P1L = P0L
P1N = P0N
x0  = np.linspace(-P0L/2,P0L/2,P0N)
y0  = np.linspace(-P0L/2,P0L/2,P0N)

def oneHole(x0,y0,holeSize):
	# Generate absorbing plane with hole in center
	return np.sign(1 - np.sign(x0**2 + y0**2 -holeSize**2 )) #Weird Heaviside function for older numpy version
	
def twoHoles(x0,y0,holeSize,sep):
	#sep = distance between the two holes
	return np.sign(1 - np.sign((x0-sep/2)**2 + y0**2 -holeSize**2 )) + np.sign(1 - np.sign((x0+sep/2)**2 + y0**2 -holeSize**2 ))

def threeHoles(x0,y0,holeSize,sep):
	# sep: distance from center of triangle to any vertex
	x1 = 0
	y1 = sep
	x2 = sep *  np.cos( np.deg2rad(30) )
	y2 = -sep * np.sin( np.deg2rad(30) )
	x3 = -x2
	y3 = y2
	h1 = np.sign(1 - np.sign((x0-x1)**2 + (y0-y1)**2 -holeSize**2 ))
	h2 = np.sign(1 - np.sign((x0-x2)**2 + (y0-y2)**2 -holeSize**2 ))
	h3 = np.sign(1 - np.sign((x0-x3)**2 + (y0-y3)**2 -holeSize**2 ))
	return h1+h2+h3

def twoSlits(x0,y0,slitWidth,slitHeight,sep):
  s1 = (-sep/2-slitWidth/2 < x0) & (x0<-sep/2+slitWidth/2) & (-slitHeight/2<y0)& (y0<slitHeight/2)
  s2 = (+sep/2-slitWidth/2 < x0) & (x0<+sep/2+slitWidth/2) & (-slitHeight/2<y0)& (y0<slitHeight/2)
  result = (s1) | (s2)
  plt.imshow(np.transpose(result))
  return result

u0 = twoHoles(x0[:,None],y0[None,:],.2,1)
#u0 = threeHoles(x0[:,None],y0[None,:],.1,1)
u0 = twoSlits(x0[:,None],y0[None,:],.2,5,1)
### Generate Results in Plane P1 ###

x1  = np.linspace(-P1L/2,P1L/2,P1N)
y1  = np.linspace(-P1L/2,P1L/2,P1N)


l= z**2+(x0[:,None,None,None] - x1[None,None,:,None])**2 + (y0[None,:,None,None]-y1[None,None,None,:])**2


u1  = (z/lam) * np.sum( u0[:,:,None,None] * (1/(k*l) - 1j) * np.exp(1j * k *l) / l**2, axis=(0,1), dtype=complex)


plt.figure()
plt.xlabel('x/lambda')
plt.ylabel('y/lambda')
plt.imshow(np.abs(np.transpose(u1)), extent=[-P0L/2,P0L/2,-P0L/2,P0L/2],interpolation='none')

plt.show()
