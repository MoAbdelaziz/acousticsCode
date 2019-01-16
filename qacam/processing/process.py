import numpy as np
import scipy.interpolate as spi
import os
import matplotlib.pyplot as plt
from datetime import date
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def plotData(data, N):
	### Plot intensity and phase data on interpolated grid 
	#N: interpolation points in each direction 
	#data:assumed 2D data with real/imaginary (columns are: x,y,real,imag)
	
	# Interpolation
	dataX = data[:,0]
	dataY = data[:,1]
	dataCoords = np.stack((dataX,dataY),axis=1)
	dataReal   = data[:,2]
	dataImag   = data[:,3]
	
	dataInt    = dataReal**2 + dataImag**2
	dataPhs    = np.arctan2(dataImag,dataReal)
	
	interpX = np.linspace(np.amin(dataX), np.amax(dataX), num=N)
	interpY = np.linspace(np.amin(dataY), np.amax(dataY), num=N)
	
	tempX,tempY = np.meshgrid(interpX,interpY)
	interpCoords = np.vstack([tempX.ravel(),tempY.ravel()]).transpose()

	interpInt = spi.griddata(dataCoords, dataInt, interpCoords, method = 'nearest')
	interpPhs = spi.griddata(dataCoords, dataPhs, interpCoords, method = 'nearest')
	print np.nanmin(dataY)
	print np.nanmax(dataY)
	
	
	# Plotting
	fig, axes = plt.subplots(2,2, sharex='col',sharey='row',figsize=(15,15))
	axes[0,0].set_title('Intensity Data')
	p1=axes[0,0].scatter(dataX,dataY, c = dataInt, cmap='inferno') # Intensity Data
	p2=axes[1,0].scatter(interpCoords[:,0],interpCoords[:,1], c = interpInt, cmap='inferno') #Interpolated
	
	divider = make_axes_locatable(axes[0,0]) # Ugly block to make colorbars work
	cax = divider.append_axes('right',size='5%',pad=.05)
	fig.colorbar(p1,cax=cax,orientation='vertical')
	divider = make_axes_locatable(axes[1,0])
	cax = divider.append_axes('right',size='5%',pad=.05)
	fig.colorbar(p2,cax=cax,orientation='vertical')
	
	axes[0,1].set_title('Phase Data')
	p1=axes[0,1].scatter(dataX,dataY, c = dataPhs, cmap='hsv') # Phase Data
	p2=axes[1,1].scatter(interpCoords[:,0],interpCoords[:,1], c = interpPhs, cmap='hsv') #Interpolated
	
	divider = make_axes_locatable(axes[0,1])
	cax = divider.append_axes('right',size='5%',pad=.05)
	fig.colorbar(p1,cax=cax,orientation='vertical')
	divider = make_axes_locatable(axes[1,1])
	cax = divider.append_axes('right',size='5%',pad=.05)
	fig.colorbar(p2,cax=cax,orientation='vertical')
	
	axes[0,0].invert_yaxis() # Invert y-axes since acam uses positive y-axis downward
	axes[1,0].invert_yaxis()
	plt.show()
	return
	
def testData():
	### Generate test data
	dataCoords = np.random.rand(10000,2)
	dataVals1   = 10*np.random.rand(10000,1)
	dataVals2   = -10*np.random.rand(10000,1)
	testData = np.concatenate((dataCoords,dataVals1,dataVals2),axis=1)
	return testData

if __name__ == "__main__":
	f = np.load('test.npy')
	plotData(f,100)

