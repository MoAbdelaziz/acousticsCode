from __future__ import division

import numpy as np
import pylab as plt
import scipy.special   as sp 
import scipy.integrate as si

# Calculating the acoustic torque of a plane standing wave on an elliptical cylinder, at a nonzero angle to its
# semi axes. Following 10.1063/1.4959071

numAlph = 20 #Number of different angles to test
numP    = 10 #Number of different b/a values to test

dataList = np.zeros((0,3))

def A(theta): # Elliptical surface function
			term1 = (np.cos(theta)/a)**2
			term2 = (np.sin(theta)/b)**2
			return (term1 + term2)**(-1/2)

def dA(theta): # Derivative of A(theta)
	numerator   = a*b*(b**2-a**2)*np.sin(2*theta)
	denominator = 2*( (b*np.cos(theta))**2 + (a*np.sin(theta))**2)**(3/2)
	return numerator/denominator

def Gamma(n,theta): # Structure function for incident wave
	factor= np.exp(1j*n*theta) 
	term1 = k * sp.jvp(n, k*A(theta))
	term2 = 1j * (n/(A(theta)**2)) * dA(theta) * sp.jv(n,k*A(theta))
	return factor * (term1 - term2)

def Upsilon(n,theta): # Structure function for scattered wave
	factor= np.exp(1j*n*theta)
	term1 = k * sp.h1vp(n, k*A(theta))
	term2 = 1j * (n/(A(theta)**2)) * dA(theta) * sp.hankel1(n,k*A(theta))
	return factor * (term1-term2)

def integrand1(n,l,theta): # Separate function for clarity
	return Gamma(n,theta)   * np.exp(-1j*l*theta)

def integrand2(n,l,theta):
	return Upsilon(n,theta) * np.exp(-1j*l*theta)

def Psi(l):
	summed = 0
	for n in range(-nmax,nmax+1):
		factor1 = np.exp(1j*n * (np.pi/2 - alpha))
		factor2 = np.exp(1j*k*h) + R * (-1)**n * np.exp(-1j*k*h)
		integral = si.simps( integrand1(n,l,thetaList), thetaList)
		summed += factor1*factor2*integral
	return (1/(2*np.pi)) * summed

def Omega(l,n):
	 # Need to double check that this is the correct way of  doing this calculation! If anything goes wrong
	 # check here first
	factor1 = np.exp(1j*n * (np.pi/2 - alpha))
	factor2 = np.exp(1j*k*h) + R * (-1)**n * np.exp(-1j*k*h)
	integral = si.simps( integrand2(n,l,thetaList), thetaList)
	return (1/(2*np.pi)) * factor1*factor2*integral 
			
			
for iAlph in np.arange(1,numAlph+1):
	for ip in np.arange(1,numP+1):
		alpha =  iAlph*1*np.pi/180
		p     =  ip*.025+1
		
		a = 1.20*10**(-6) # minor axis, meters
		#p = 1.1 #scaling factor for b relative to a
		b = p*a # major axis
		k = 8378          # wavenumber, reciprocal meters (2 MHz in water)
		E0 = 10 #characteristic acoustic energy density (J/m^3)
		eta = 10**-3 #viscosity, Pa*s

		h = np.pi/(4*k)   # axial distance between cylinder center and plane wave source (still need to fully understand this)
		#alpha = 10 * np.pi/180 # angle incident wave makes with minor axis
		R = 1 # 0 for travelling wave, 1 for standing wave, intermediate for quasi-standing

		nmax = 10 # For sums over all n, calculate from -nmax to nmax
		lmax = 10 # Likewise for l

		thetaList =np.linspace(0,2*np.pi,num=1000)  # Theta values over which to do integration, plotting, etc

		# Construct the matrix of Omega(l,n)
		OmegaMat = np.zeros((2*lmax +1 , 2*nmax +1),dtype='complex')
		for n in range (-nmax,nmax+1):
			for l in range(-lmax,lmax+1):
				OmegaMat[l+lmax,n+nmax]=Omega(l,n) # need the +lmax, +lmax since Python starts indexing at 0, not negative values

		OmegaMatInv = np.linalg.inv(OmegaMat)
	
		PsiVec = np.zeros(2*lmax+1,dtype='complex') # vector of Psi(l)
		for l in range(-lmax,lmax+1):
			PsiVec[l+lmax] = Psi(l)
	
		# Find scattering coefficients from matrix inversion and multiplication
		CVec = -np.einsum('nl,l->n',OmegaMatInv,PsiVec)

		# Calculate the non-dimensional radiation torque
		summed=0
		for n in range(-nmax,nmax+1):
			factor1 = n*(  (R**2+1) + 2*(-1)**n * R * np.cos(2*k*h) ) 
			alphan = np.real(CVec[n+nmax]) 
			betan  = np.imag(CVec[n+nmax])
			factor2 = alphan+alphan**2+betan**2
			summed += factor1*factor2
	
		nonDimTorque = -4/(np.pi * (k*b)**2 * (1+R)**2) * summed # non dimensional radiation torque

		torquePerLength = nonDimTorque * E0 * np.pi * b**2 


		rotRate = torquePerLength*2*a /(8*np.pi*eta*a**3) #multiply by 2a to approximate torque on one droplet, then find the
														  #rotation rate as usual for spheres
		
		dataList = np.append(dataList,[[alpha,p,rotRate]],axis=0)
	
	

np.save('data.npy',dataList)
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	

