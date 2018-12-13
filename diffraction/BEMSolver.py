from __future__ import division
import numpy as np
import pylab as plt
### This does the evaluation of the pressure in the region given the pressure on the boundaries, but needs the matrix solver step first
### to correctly find the pressure on the boundaries

### Initialize 

### Define geometry and boundary conditions

# Simple case: a rigid square, plane wave incidence
def rigidSquare(L,N):
	#L: side length (centered at 0,0,0, extends in xy plane)
	#N: number of points per side to resolve
	xc = np.linspace(-L/2,L/2,N)
	yc = np.linspace(-L/2,L/2,N)
	xg,yg = np.meshgrid(xc,yc)
	points = np.c_[np.ravel(xg),np.ravel(yg)]
	numPoints = np.shape(points)[0]
	n  = np.asarray([0,0,-1]) # Normal vector on square is just in negative z direction
	dA = L**2/N**2*np.ones((numPoints,numPoints)) #area per point
	return points, n, dA, numPoints
	
surfacePoints,n,dA, numPoints = rigidSquare(1,10)

## Create sources
def pointSource(lam, x0,y0,z0,p0 , x,y,z):
	#x0,y0,z0: source location
	#p0: source strength
	#x,y,z: evaluation point

	#lam: wavelength of sound
	k = (2*np.pi)/lam
	xd = x-x0
	yd = y-y0
	zd = z-z0
	
	r = np.sqrt( xd**2 + yd**2 + zd**2 ) 
	
	#p: pressure at x,y,z
	p = np.exp(-1j * k*r) / (4*np.pi*r)  
	
	return p

### Specify regions of evaluation

# Evaluate along z-axis
def evalZAxis(zi,zf,N):
	return np.linspace(zi,zf,N)

evalPoints = evalZAxis(-2,2,100)

### Calculate Kirchoff integrand at each area element
def green(lam, xb,yb,zb , x,y,z):
	#b coords: coordinates on boundary
	#x,y,z: evaluation point
	k = (2*np.pi)/lam
	xd = x-xb
	yd = y-yb
	zd = z-zb
	r = np.sqrt( xd**2 + yd**2 + zd**2 ) 
	return np.exp(-1j * k * r)/(4*np.pi*r)

def greenD(lam, xb,yb,zb, x,y,z , nx,ny,nz):
	# Need local unit normal vector to surface, (nx,ny,nz) to find dG/dn
	k = (2*np.pi)/lam
	xd = x-xb
	yd = y-yb
	zd = z-zb
	r = np.sqrt( xd**2 + yd**2 + zd**2 ) 
	rdn = (xd*nx + yd*ny + zd*nz)/(r) #r dot n, normalized by r (n should already be normalized when input)
	dGdr = -np.exp(-1j * k * r) * (1+ 1j * k * r) /(r * np.pi*r)
	return dGdr * rdn

def boundaryField():
	# Matrix solver for pressure field on scattering surface
	bMat = np.zeros((numPoints,numPoints))
	
	return
    
def kirchIntegrand(lam, xb,yb,zb, x,y,z, nx,ny,nz, dA,p, BC = 'rigid'):
	#Evaluates the Kirchoff integrand at one area element
	#BC = 'rigid'; take dp/dn = 0
	#BC = 'soft' ; take p = 0 
	#Can add continuum of rigid-soft at some point if needed
	#dA: amount of area per point
	#p: p at xb,yb,zb from source
	if BC == 'rigid':
		result = -dA * greenD(lam,xb,yb,zb , x,y,z, nx,ny,nz) * p
	if BC == 'soft':
		result = 0 #Not accounted for yet
	return result
	
# Test by evaluating at just one point
#kirchArray = kirchIntegrand(lam, squarePoints[:,0], squarePoints[:,1],1, 0,0,.5,  n[0],n[1],n[2], dA, pointSource(lam,0,0,0,1,squarePoints[:,0], squarePoints[:,1],1),BC = 'rigid')


### Display results
