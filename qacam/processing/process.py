import numpy as np
import scipy.interpolate as spi
import os
import matplotlib.pyplot as plt
from datetime import date

def saveData(data):
	### Save the current run, automatically dating and numbering the file
	i = 1
	today = str(date.today())
	filename =  today+"_N%s.npy"
	
	while os.path.exists(filename % i):
		i +=1
	
	np.save(filename % i, data)
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

	interpInt = spi.griddata(dataCoords, dataInt, interpCoords, method = 'cubic')
	interpPhs = spi.griddata(dataCoords, dataPhs, interpCoords, method = 'cubic')
	
	# Plotting
	fig, axes = plt.subplots(2,2, sharex='col',sharey='row')
	axes[0,0].set_title('Intensity Data')
	axes[0,0].scatter(dataX,dataY, c = dataInt) # Intensity Data
	axes[1,0].scatter(interpCoords[:,0],interpCoords[:,1], c = interpInt) #Interpolated
	
	axes[0,1].set_title('Phase Data')
	axes[0,1].scatter(dataX,dataY, c = dataPhs) # Phase Data
	axes[1,1].scatter(interpCoords[:,0],interpCoords[:,1], c = interpPhs) #Interpolated
	plt.show()
	return
	
def testData():
	### Generate test data
	dataCoords = np.random.rand(10000,2)
	dataVals1   = 10*np.random.rand(10000,1)
	dataVals2   = -10*np.random.rand(10000,1)
	testData = np.concatenate((dataCoords,dataVals1,dataVals2),axis=1)
	return testData

