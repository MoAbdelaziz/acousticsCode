import numpy as np
import scipy.interpolate as spi
import os
import matplotlib.pyplot as plt
from datetime import date
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

def saveData(data):
	### Save the current run, automatically dating and numbering the file
	i = 1
	today = str(date.today())
	direc = 'data/'
	filename =  today+"_N%s.npy"
	
	while os.path.exists(direc+filename % i):
		i +=1
	
	np.save(direc+filename % i, data)
	return

def plotData(data, N, levels = 100, intensityScale = 'sqrt', plot=True):
	### Plot intensity and phase data on interpolated grid 
	#N: interpolation points in each direction 
	#data:assumed 2D data with real/imaginary (columns are: x,y,real,imag)
	#intensityScale = 
			# 'linear'; default display
			# 'log'   ; take log of intensity before plotting 
	# Interpolation
	dataX = data[:,0]
	dataY = data[:,1]
	dataCoords = np.stack((dataX,dataY),axis=1)
	dataReal   = data[:,2]
	dataImag   = data[:,3]
	
	dataInt = dataReal**2 + dataImag**2

	if intensityScale =='linear':
		dataInt = dataReal**2 + dataImag**2

	if intensityScale=='log':
		dataInt = np.log(dataReal**2+dataImag**2)

	if intensityScale=='sqrt':
		 dataInt = np.sqrt(dataReal**2+dataImag**2)

	dataPhs    = np.arctan2(dataImag,dataReal)
	
	interpX = np.linspace(np.amin(dataX), np.amax(dataX), num=N)
	interpY = np.linspace(np.amin(dataY), np.amax(dataY), num=N)
	
	tempX,tempY = np.meshgrid(interpX,interpY)
	interpCoords = np.vstack([tempX.ravel(),tempY.ravel()]).transpose()

	interpInt = spi.griddata(dataCoords, dataInt, interpCoords, method = 'nearest')
	interpPhs = spi.griddata(dataCoords, dataPhs, interpCoords, method = 'nearest')

	
	if plot==True:
	  # Plotting
	  
	  
	  fig, axes = plt.subplots(3,2, sharex='col',sharey='row',figsize=(15,15))
	  ax3D = plt.subplot(325,projection='3d')
	  # 2D contours of intensity and pahse
	  axes[0,0].set_title('Intensity Data')
	  p1=axes[0,0].scatter(dataX,dataY, c = dataInt, cmap='inferno') # Intensity Data
	  #p2=axes[1,0].scatter(interpCoords[:,0],interpCoords[:,1], c = interpInt, cmap='inferno') #Interpolated
	  p2 = axes[1,0].contourf(interpCoords[:,0].reshape(N,N),interpCoords[:,1].reshape(N,N),  interpInt.reshape(N,N), levels,cmap='inferno') #Interpolated
	  
	  # 3D surface plots
	  p3 = ax3D.plot_surface(interpCoords[:,0].reshape(N,N),interpCoords[:,1].reshape(N,N),  interpInt.reshape(N,N), cmap='inferno')
	  
	  divider = make_axes_locatable(axes[0,0]) # Ugly block to make colorbars work
	  cax = divider.append_axes('right',size='5%',pad=.05)
	  fig.colorbar(p1,cax=cax,orientation='vertical')
	  divider = make_axes_locatable(axes[1,0])
	  cax = divider.append_axes('right',size='5%',pad=.05)
	  fig.colorbar(p2,cax=cax,orientation='vertical')
	  
	  axes[0,1].set_title('Phase Data')
	  p1=axes[0,1].scatter(dataX,dataY, c = dataPhs, cmap='hsv') # Phase Data
	  #p2=axes[1,1].scatter(interpCoords[:,0],interpCoords[:,1], c = interpPhs, cmap='hsv') #Interpolated
	  p2 = axes[1,1].contourf(interpCoords[:,0].reshape(N,N),interpCoords[:,1].reshape(N,N),  interpPhs.reshape(N,N),levels, cmap='hsv') #Interpolated
	  
	  divider = make_axes_locatable(axes[0,1])
	  cax = divider.append_axes('right',size='5%',pad=.05)
	  fig.colorbar(p1,cax=cax,orientation='vertical')
	  divider = make_axes_locatable(axes[1,1])
	  cax = divider.append_axes('right',size='5%',pad=.05)
	  fig.colorbar(p2,cax=cax,orientation='vertical')
	  

	  
	  axes[0,0].invert_yaxis() # Invert y-axes since acam uses positive y-axis downward
	  axes[1,0].invert_yaxis()
	  
	  plt.show()
	  
	# Other calculations
	# Total power through scan section (intensity per pixel summed up)
	xRange = np.ptp(interpX)
	yRange = np.ptp(interpY)
	pixelArea = (xRange * yRange)/(N**2)
	totalPower = np.sum(interpInt) * pixelArea # note units: chosen intensity scale * cm^2
	print('Total power in scan range: ' + np.array2string(totalPower) )
	return interpCoords[:,0].reshape(N,N),interpCoords[:,1].reshape(N,N),  interpInt.reshape(N,N), interpPhs.reshape(N,N)
	
def testData():
	### Generate test data
	dataCoords = np.random.rand(10000,2)
	dataVals1   = 10*np.random.rand(10000,1)
	dataVals2   = -10*np.random.rand(10000,1)
	testData = np.concatenate((dataCoords,dataVals1,dataVals2),axis=1)
	return testData

if __name__ == "__main__":
<<<<<<< HEAD
	f = np.load('2019-06-07_N6.npy')
	plotData(f,200,intensityScale='linear')
=======
	f = np.load('2019-01-28_N8.npy')
	plotData(f,100)
>>>>>>> d6d7cb4c08c4a4369767d03b83bf9df79607c0eb

