from __future__ import division 
#import matplotlib
#matplotlib.use("Agg")
import numpy as np
import pylab as plt

from matplotlib import animation

writer = animation.ImageMagickFileWriter()
#writer = animation.writers['imagemagick']
#writer = Writer(fps=15,metadata=dict(artist='Me'),bitrate=1800)
#np.random.seed(123456)

# Account for forces:
# 1) Acoustic interaction between two particles, which is attractive when both are in the 
# trapping plane, but repulsive for other arrangements
# 2) Advection rotation due to particle rotation
# 3) Primary trapping force
# 4) Buoyancy + gravity
# 5) Diffusion
# 6) Excluded volume (needs improvement; currently calculating only for 2 particles at a time, should check ALL particles at once and correct them)

Np = 40# Number of particles
Nt = 10000 # Number of time steps
dt = .01 # currently in seconds... estimate for typical time taken to travel one particle radius is ________
interval = 2 #ms between frames for animations

a = 1.6*10**(-6)# particle radius, everything in SI units for now
rhom = 1000
rhop = 1180 
etam = 10**(-3)
cm = 1500
cp = 1350
Omega =20
E0 = 10 # Acoustic energy density 
f1 = 2*(rhop-rhom)/(2*rhop+rhom)
f = 2*10**6
omega = 2 *np.pi * f
k = omega/cm
betap = 1/(rhop * cp**2)
betam = 1/(rhom * cm**2)
G = 1 -  betap/betam + f1*(3/2) # Acoustic contrast factor
g = 9.81 # gravitational accel
kB = 1.38*10**(-23)
T = 290 # Room temp

lims = 30*a # Box ranges from -lims to lims in all 3 directions , so lims is also the cutoff distance (half box size) for the interaction (periodic BCs). 

contact = 2*a # 2*a default, closest two particle centers are allowed to approach

pos = np.zeros((Np,Nt+1,3)) # Position array
#z: plane wave propagation direction
#y: torque axis
#x: Other axis in levitation plane


# Initial distribution of particles
def chain(Np,sep,noise):
	# Chain aligned more along torque (y) in trapping plane (xz), slightly offset in x
	for n in range(Np):
		pos[n,0,0] = np.random.rand()*noise
		pos[n,0,1] = -(Np)*sep/2 + (Np)*sep*n/Np 
		pos[n,0,2] = 0 
	ics = 'chain'
	return pos,ics
def far(Np,sep,noise):
	# Particles far from each other, but near trapping plane
	for n in range(Np):
		pos[n,0,0]=sep+np.random.rand()*noise
		pos[n,0,1]=sep+np.random.rand()*noise
		pos[n,0,2]=np.random.rand()*noise/10
	ics = 'far'
	return pos,ics
def grid(Np,sep,noise):
	 #grid in xy with no randomness
	for n in range(Np):
		pos[n,0,0] = -(Np-1)*sep/2 + (Np-1)*sep*n/Np 
		pos[n,0,1] = -(Np-1)*sep/2 + (Np-1)*sep*n/Np 
		pos[n,0,2] = sep
	ics = 'grid'
	return pos,ics
def rand(Np,noise):
	# random
	for n in range(Np):
		for coord in [0,1,2]:
			pos[n,0,coord] = (np.random.rand()-0.5)*noise
	#pos[n,0,2] = 0 #initialize in trapping plane
	ics = 'rand'
	return pos,ics
def aligned(Np,xs,ys,zs):
	for n in range(Np):
		pos[n,0,:]=[xs*n*a,ys*n*a,zs*n*a]
	ics = 'aligned'
	return pos,ics
 
#pos,ics  = chain(Np, 5*a, 15*a )
#pos,ics  = far  (Np, 0*a, 20*a )
#pos,ics  = grid (Np, 3*a, 0.1*a)
pos,ics  = rand (Np,      10*a )

filename = 'Np'+str(Np)+'Nt'+str(Nt)+'dt'+str(dt)+'ic'+ics+'O'+str(Omega)+'.npy'
	
#def Bjerknes(dx,dy,dz,h): ## Old version, this one might be either entirely or partially wrong, will rewrite in a cleaner way
	## Calculate Bjerknes force DUE to particle i at height h above trapping plane (z=0) on particle j.
					## Acoustic interaction force
				## First code just the 1/(kr)^3 term
	#prefac = np.nan_to_num(a**6 * E0 * f1**2 * np.pi * np.cos(h*k)/(2 * drho * r**7))
				
	#Fsb0 =np.nan_to_num( r**2 * np.cos(k*(dz-h)) * (k*r * drho * (1+3*np.cos(2*theta)) * np.sin(k*r) + 3 * np.cos(k*r)*(drho + 3 *drho * np.cos(2*theta) + 2 * dz * np.sin(2*theta))))
						
	#Fsbx = Fsb0 *dx
	#Fsby = Fsb0 *dy
	
	#Fsbz = np.nan_to_num(r**2 * drho * ( (dz*np.cos(k*(dz-h))*(3*np.cos(k*r)*(r**2-4*drho**2+3*r**2*np.cos(2*theta)) + k*r**3
			#*(1+3*np.cos(2*theta))*np.sin(k*r)))/(r**2) + k*r**2*np.cos(k*r)*(1+3*np.cos(2*theta))*np.sin(k*(dz-h))))
	
	#Fsb = prefac*np.array([Fsbx,Fsby,Fsbz])
	
	#Vsb = Fsb/(6*np.pi*etam*a) 
	
	#return Vsb

def Bjerknes(x,y,z,h): #New version, should be correct
	# Bjerknes force on particle j due to i. i is at the origin, and h is the distance from the pressure node to particle i.
	# Only the first term is included (1/(kr)**3 order) because it's the only one that matters for ka<<1
	# x,y,z distances from i to j
	
	prefactor = np.pi*E0*k**3*a**6 
	
	# Calculate r and theta
	r     = np.sqrt(x**2+y**2+z**2)
	theta = np.arctan2(np.sqrt(x**2+y**2),z)
	
	# Calculate the force components in spherical first
	rComp1 = (3*f1**2*np.cos(h*k)*np.cos(k*r)*(1+3*np.cos(2*theta))*(np.cos(k*(r*np.cos(theta)-h))))/(2*k**3*r**4)
	rComp2 = (f1**2*np.cos(h*k)*(1+3*np.cos(2*theta))*np.cos(k*(r*np.cos(theta)-h))*np.sin(k*r))/(2*k**2*r**3)
	rComp3 = (f1**2*np.cos(h*k)*np.cos(k*r)*np.cos(theta)*(1+3*np.cos(2*theta))*np.sin(k*(r*np.cos(theta)-h)))/(2*k**2*r**3)
	rComp = rComp1 + rComp2 + rComp3
	thetaComp1 = (3*f1**2*np.cos(h*k)*np.cos(k*r)*np.cos(k*(r*np.cos(theta)-h))*np.sin(2*theta))/(k**3*r**4)
	thetaComp2 = (-f1**2*np.cos(h*k)*np.cos(k*r)*(1+3*np.cos(2*theta))*np.sin(theta)*np.sin(k*(r*np.cos(theta)-h)))/(2*k**2*r**3)
	thetaComp =  thetaComp1 + thetaComp2
	
	# Unit vector definitions
	xh = np.array([1,0,0])
	yh = np.array([0,1,0])
	zh = np.array([0,0,1])
	rh = (1/r)*(x*xh + y*yh +z*zh)

	if x==0 and y==0:
		print 'test'
		thetah=0
	else: 
		thetah = (1/(r*np.sqrt(x**2+y**2)))*( z*(x*xh+y*yh) -(x**2+y**2)*zh)
		
	Fsb = prefactor*(rComp*rh + thetaComp*thetah)
	return Fsb/(6*np.pi*etam*a)

def hydroInt():
	# Calculate first order hydrodynamic interactions
	# Find flow at j due to total force at i
	return
def Fz(z):
	# Primary radiation (trapping) force
	return -4*np.pi*a**3 * E0*k*G*np.sin(2*k*z)

def periodicBC():
	# Apply periodic BCs to positions of particles that exited in xy range
	for n in range(Np):
		for coord in [0,1]:
			if pos[n,ti+1,coord] < -lims: # If gone in negative direction, add total box size (2 lims)
				pos[n,ti+1,coord]+= 2*lims
			if pos[n,ti+1,coord] >  lims: # If gone in positive direction, subtract total box size (2 lims)
				pos[n,ti+1,coord]-= 2*lims
	return 
      
def excludedVolume():
	# Check all pairs for clipping and separate them
	for i in range(Np):
		for j in range(i+1,Np):
			dr = pos[j,ti,:] - pos[i,ti,:]
			r = np.linalg.norm(dr)
			# Deal with the unphysical r<2*a case
			if r<contact:
				pos[j,ti+1,:] += (contact-r)*(dr/r) /2 #adjust starting future positions so spheres are just touching, giving half the difference to j, and half to i
				pos[i,ti+1,:] -= (contact-r)*(dr/r) /2
				dr = (dr/r) * contact #make the separation act like 2a for any smaller separations (is this ok?)
				r  = contact #correct magnitude as well
				periodicBC() # If this pushing process sent any out of the box, correct it
				
for ti in range(Nt):
	t = ti*dt
	pos[:,ti+1,:] = pos[:,ti,:]
	for i in range(Np):
		for j in range(i,Np): #Avoid double counting and just be sure to calculate reciprocal force as well 
			if i!=j:
			  
				# Calculate x,y,z differences
				dr = pos[j,ti,:] - pos[i,ti,:]
				r = np.linalg.norm(dr)
				
				dx = dr[0]
				dy = dr[1]
				dz = dr[2]
				
				# Apply Periodic BCs to the interaction (not in z, b/c of standing wave)
				if np.abs(dx) > lims:
					dx = dx - 2*lims * np.sign(dx)
				if np.abs(dy) > lims:
					dy = dy - 2*lims * np.sign(dy)
				dr = np.asarray([dx,dy,dz])
				r = np.linalg.norm(dr) #re-update r,dr (inefficient?)
				
				drho = np.sqrt(dx**2+dy**2)
				
				hi = pos[i,ti,2] # if z=0 is the trapping plane, h is where the potential-generating sphere is located
				                # along z ( I think ).
				hj = pos[j,ti,2]
				
				# Calculate other coordinates
				
				theta = np.arctan2(np.sqrt(dx**2+dy**2),dz)
				psi = np.arctan2(np.sqrt(dz**2+dx**2),dy)
				
				# Flow velocity corresponding to advected rotational flow
				Vadv = np.nan_to_num(  (a**3)/(r**2) * Omega * np.sin(psi)*(1/np.sqrt(dz**2+dx**2))*(-dx*np.array([0,0,1]) + dz *np.array([1,0,0]))  )
				
				# Contribution from Bjerknes force
				Vsbj = Bjerknes( dx, dy, dz,hi) #Flow at j depends on height of i, anFadvd coordinates from i to j
				Vsbi = Bjerknes(-dx,-dy,-dz,hj) # Vice verse 
				

				pos[j,ti+1,:] += ( Vadv + Vsbj )*dt
				pos[i,ti+1,:] += (-Vadv + Vsbi )*dt #negate Vadv because it's calculated for i on j, 
				
		# Contribution from primary radiation force (must be in the higher loop because it's not an interaction)
		Vzi = Fz(pos[i,ti,2])/(6*np.pi*etam*a)
		pos[i,ti+1,2] += Vzi*dt
		
		# Gravity and buoyancy...
		Vgb = (4/3)*np.pi*a**3 *(rhom-rhop) * g/(6*np.pi*etam*a)
		pos[i,ti+1,2] += Vgb*dt
		
		# Diffusion!
		xi = np.random.normal(0,1,3) # Pick 3 values (x,y,z) from Gaussian centered at 0, unit variance
		pos[i,ti+1,:] += np.sqrt(2*kB*T/(6*np.pi*etam*a)) * np.sqrt(dt) * xi
		
	if (ti/Nt)%0.1 ==0:
		print ti/Nt
	periodicBC() # Update positions if any fell out of the box
	excludedVolume()
	
	
# Save all position data

np.save(filename, pos)
  
  
  
# Plot test results
lims /=a
pos /= a # Normalize positions by particle radius

fig = plt.figure() #XZ Plots (torque is in y)
ax = plt.axes(xlim=(-lims,lims),ylim=(-lims,lims))
s = ((ax.get_window_extent().width  / (2*lims) *20/fig.dpi) ** 2) #marker sizes scaled for plot (not scatter), trial and error

for n in range(Np):
	plt.plot(pos[n,:,0],pos[n,:,2],alpha=0.5)
  
plt.plot(np.linspace(-lims,lims),0*np.linspace(-lims,lims),'k-') # Trapping plane
plt.scatter(pos[:,0,0],pos[:,0,2],s=s) # Highlight starting points
plt.scatter(pos[:,-1,0],pos[:,-1,2],marker='s',s=s) # Highlight ending points
plt.show()

plt.figure() #XY Plots (trapping plane)
plt.xlim((-lims,lims))
plt.ylim((-lims,lims))
for n in range(0,Np):
	plt.plot(pos[n,:,0],pos[n,:,1],alpha=0.5)
plt.scatter(pos[:,0,0],pos[:,0,1],s=s) # Highlight starting points
plt.scatter(pos[:,-1,0],pos[:,-1,1],marker='s',s=s) # Highlight ending points
plt.show()


## New method with circle
# Set up figure for animation (XZ)
fig = plt.figure()
ax = plt.axes(xlim=(-lims,lims),ylim=(-lims,lims))
ax.set_aspect(1)
title = ax.text(0,lims, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, ha="center")
plt.xlabel('x')
plt.ylabel('z')
def init():
	return []
      
# Animation function, will be called sequentially
def animate(i):
	#title.set_text(str(i*dt))
	patches=[]
	for n in range(Np):
		x = pos[n,i,0]
		z = pos[n,i,2]

		patches.append(ax.add_patch(plt.Circle((x,z),radius=1,alpha=0.5)))

	return patches
# Call the animator
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Nt,interval= interval,blit=True)
plt.show()
#anim.save('xz.mp4',writer=writer)

# Set up figure for animation (XY)
fig = plt.figure()
ax = plt.axes(xlim=(-lims,lims),ylim=(-lims,lims))
ax.set_aspect(1)
title = ax.text(0,lims, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
plt.xlabel('x')
plt.ylabel('y')
def init():
	return []
      
# Animation function, will be called sequentially
def animate(i):
	#title.set_text(str(i*dt)+' s')
	patches=[]
	for n in range(Np):
		x = pos[n,i,0]
		y = pos[n,i,1]
		patches.append(ax.add_patch(plt.Circle((x,y),radius=1,alpha=0.5)))

	return patches
# Call the animator
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Nt,interval= interval,blit=True)
plt.show()
#anim.save('xy.gif',writer=writer)
	
# Other useful analytics
fig,axes = plt.subplots(3,1 , sharex=True)
for n in range(Np):
	for coord in[0,1,2]:
		axes[coord].plot(np.arange(Nt+1)*dt,pos[n,:,coord],alpha=0.5)
plt.show()







## Old method with plt.plot
## Set up figure for animation (XZ)
#fig = plt.figure()
#ax = plt.axes(xlim=(-lims,lims),ylim=(-lims,lims))
#line, = ax.plot([],[],markersize=s,linestyle='',marker='.',alpha=0.5)
#plt.xlabel('x')
#plt.ylabel('z')
#def init():
	#line.set_data([],[])
	#return line,
      
## Animation function, will be called sequentially
#def animate(i):
	#x = pos[:,i,0]
	#z = pos[:,i,2]
	#plt.title(str(i*dt)+' s')
	#line.set_data(x,z)

	#return line,
## Call the animator
#anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Nt,interval= interval,blit=False)
#plt.show()

## Set up figure for animation (XY)
#fig = plt.figure()
#ax = plt.axes(xlim=(-lims,lims),ylim=(-lims,lims))
#line, = ax.plot([],[],markersize=s,linestyle='',marker='.',alpha=0.5)
#plt.xlabel('x')
#plt.ylabel('y')
#def init():
	#line.set_data([],[])
	#return line,
      
## Animation function, will be called sequentially
#def animate(i):
	#x = pos[:,i,0]
	#y = pos[:,i,1]
	#plt.title(str(i*dt)+' s')
	#line.set_data(x,y)
	#return line,
## Call the animator
#anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Nt,interval= interval,blit=False)
#plt.show()
	
## Other useful analytics
#fig,axes = plt.subplots(3,1 , sharex=True)
#for n in range(Np):
	#for coord in[0,1,2]:
		#axes[coord].plot(np.arange(Nt+1)*dt,pos[n,:,coord],alpha=0.5)
#plt.show()