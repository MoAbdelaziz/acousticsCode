from __future__ import division 
#import matplotlib
#matplotlib.use("Agg")
import numpy as np
import pylab as plt

from matplotlib import animation
import re

writer = animation.ImageMagickFileWriter()

a = 1.6*10**(-6)# particle radius, everything in SI units for now
lims = 20*a
Np=20
Nt = 10000 
dt = .01
interval = 2 #ms between frames

pos = np.load('Np20Nt10000dt.01rand.npy')
  
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







