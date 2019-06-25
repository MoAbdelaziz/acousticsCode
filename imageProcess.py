
# coding: utf-8

# In[4]:


#          
# Functions for taking data from 3D measurement of amplitude and phase, interpolating it onto a grid, and producing relevant plots 
#
import numpy as np
from matplotlib import pylab as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
# Generate sample data for test purposes


def fakeData(x,y,z,option):
    if option ==1:
        return np.sin(x)
    if option ==2:
        return np.cos(y)

    
size = 10000
x = np.random.standard_normal(size)
y = np.random.standard_normal(size)
z = np.random.standard_normal(size)

data1 = fakeData(x,y,z,1)
data2 = fakeData(x,y,z,2)

dataArray = np.stack((x,y,z,data1,data2),axis=0).T





def process(data, actual=True, interp = True,xSamples = 25,ySamples=25,zSamples=3,  interpMode = 'linear', xContour = True, xSlice=1 ,zContour = True,zSlice = 1, showAllInterp = False):
    # Assume first 3 columns are x,y,z coords, remaining 2 are amplitude,phase
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    amplitude = data[:,3]
    intensity = np.multiply(amplitude,amplitude)
    phase = data[:,4]
    
    if actual == True:
        # Just plot all the data as a scatterplot, with values given by color
        fig = plt.figure(figsize=(8,8))
        
        unqX = np.unique(x).size
        unqY = np.unique(y).size
        unqZ = np.unique(z).size # Check how many unique values of each coordinate there are to determine the best plot
        
        if unqZ ==1:
            ax1 = fig.add_subplot(111)
            p1 = ax1.scatter(x, y, c=intensity, cmap=plt.jet())
            fig.colorbar(p1)
        elif unqX ==1 and unqY ==1:
            ax1 = fig.add_subplot(111)
            p1 = ax1.scatter(z, intensity)
        else:   
            ax1 = fig.add_subplot(111,projection = '3d')
            p1 = ax1.scatter(x, y,z, c=intensity, cmap=plt.jet())
            fig.colorbar(p1)
       # fs = 18 #Global font size
       # ax1.set_title('Amplitude',fontsize=fs)
       # ax2.set_title('Phase',fontsize=fs)
       # ax1.set_xlabel('x',fontsize=fs), ax1.set_ylabel('y', fontsize=fs), ax1.set_zlabel('z',fontsize=fs) 
       # ax2.set_xlabel('x',fontsize=fs), ax2.set_ylabel('y', fontsize=fs), ax2.set_zlabel('z',fontsize=fs)
        
        
        
        
        fig = plt.figure(figsize=(8,8))
        
        if unqZ ==1:
            ax2 = fig.add_subplot(111)
            p2 = ax2.scatter(x, y, c=phase, cmap=plt.jet())
            fig.colorbar(p2)
        elif unqX ==1 and unqY ==1:
            ax2 = fig.add_subplot(111)
            p2 = ax2.scatter(z, phase)
        else: 
            ax2 = fig.add_subplot(111,projection = '3d')
            p2 = ax2.scatter(x, y, z, c=phase, cmap=plt.jet())
            fig.colorbar(p2)
        
    
    if interp == True: # Perform some interpolation
        #Create an xyz grid based on chosen samples in each direction
        xi = np.linspace(min(x),max(x),xSamples)
        yi = np.linspace(min(y),max(y),ySamples)
        zi = np.linspace(min(z),max(z),zSamples)
        
        if interpMode == 'linear': #linear interpolation of data on the uniform grid
            interp1 = LinearNDInterpolator((x,y,z),amplitude).__call__(xi[:,None,None],yi[None,:,None],zi[None,None,:])
            interp2 = LinearNDInterpolator((x,y,z),phase).__call__(xi[:,None,None],yi[None,:,None],zi[None,None,:])
            
        
        elif interpMode == 'nearest': #nearest neighbor interpolation of data on  the uniform grid
            interp1 = NearestNDInterpolator((x,y,z),amplitude).__call__(xi[:,None,None],yi[None,:,None],zi[None,None,:])
            interp2 = NearestNDInterpolator((x,y,z),phase).__call__(xi[:,None,None],yi[None,:,None],zi[None,None,:])
        else:
            print "Invalid interpolation mode, use either linear or nearest"
            return
        
        # This is probably equivalent to just using np.meshgrid
        xd = np.arange(interp1.shape[0])[:,None,None]
        yd = np.arange(interp1.shape[1])[None,:,None]
        zd = np.arange(interp1.shape[2])[None,None,:]
        xd, yd, zd = np.broadcast_arrays(xd, yd, zd)
        
        c1= interp1.ravel()
        c1 =(c1 - np.nanmin(c1))/(np.nanmax(c1)-np.nanmin(c1))
        
        c2= interp2.ravel()
        c2 =(c2 - np.nanmin(c2))/(np.nanmax(c2)-np.nanmin(c2))

        if showAllInterp == True:
            # Do the plotting in a single call.
            fig = plt.figure(figsize=(8,8))
            ax = fig.gca(projection='3d')
            ax.scatter(xd.ravel(),
                       yd.ravel(),
                       zd.ravel(),
                       c=c1,
                      alpha=.3)

            fig = plt.figure(figsize=(8,8))
            ax = fig.gca(projection='3d')
            ax.scatter(xd.ravel(),
                       yd.ravel(),
                       zd.ravel(),
                       c=c2,
                      alpha=.3)
        
        if zContour == True: # Plot a 2d contour at chosen zSlice
            
            
            kept = zd.ravel()== zSlice # keep only the values at positions equal to chosen zSlice

            fig = plt.figure(figsize =(8,8))
            ax = fig.gca()
            ax.contourf(np.unique(xd.ravel()[kept]),
                       np.unique(yd.ravel()[kept]),
                       np.reshape(c1[kept], (xSamples,ySamples)).T)
            
            fig = plt.figure(figsize =(8,8))
            ax = fig.gca()
            ax.contourf(np.unique(xd.ravel()[kept]),
                       np.unique(yd.ravel()[kept]),
                       np.reshape(c2[kept], (xSamples,ySamples)).T)
            
        if xContour == True: # Plot a 2d contour at chosen xSlice
            
            
            kept = xd.ravel()== xSlice # keep only the values at positions equal to chosen zSlice

            fig = plt.figure(figsize =(8,8))
            ax = fig.gca()
            ax.contourf(np.unique(yd.ravel()[kept]),
                       np.unique(zd.ravel()[kept]),
                       np.reshape(c1[kept], (ySamples,zSamples)).T)
            
            fig = plt.figure(figsize =(8,8))
            ax = fig.gca()
            ax.contourf(np.unique(yd.ravel()[kept]),
                       np.unique(zd.ravel()[kept]),
                       np.reshape(c2[kept], (ySamples,zSamples)).T)
    return

process(dataArray,actual = True, interp = True, showAllInterp=False,zSamples=5,interpMode ='nearest', zSlice = 2, xSamples = 1000, ySamples = 1000,xSlice = 500)


# In[6]:


1==2 or 1==1 and 1==1

