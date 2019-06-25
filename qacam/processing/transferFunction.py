###
### Divide an image produced by a grating by one produced with no grating to see the transfer properties of the grating
### 

import numpy as np
import matplotlib.pyplot as plt
from process import plotData
from mpl_toolkits.axes_grid1 import make_axes_locatable


if __name__ == "__main__":
  # Run by putting 'background.py' and 'image.py' in the same folder as this script (or just change the filenames below)
  N = 100 # pixels per side for interpolation
  levels =300 # number of countours for plotting
  intensityScale = 'sqrt' # scale for each data set
  divideMethod   = 'linear' #log, linear, sqrt, diff; how to divide background by input
  
  
  # Import grid-interpolated coordinates, intensity, and phase data from each data set
  backgroundData = np.load('2019-01-28_N9.npy')
  imageData      = np.load('2019-01-28_N8.npy'     )
  print np.sqrt(  np.shape(imageData)[0] ) # Estimate of original data resolution (pixels per side)
  #b: background, i: image, x,y: coords, I: intensity, P:phase
  bx,by,bI,bP = plotData(backgroundData, N, intensityScale=intensityScale, plot=False)
  bI+=.1 #temporary!!!! artificially get rid of issues with small background values which lead to huge intensity quotients
  ix,iy,iI,iP = plotData(imageData    , N, intensityScale=intensityScale, plot=False)
  #Ideally, bx and by are similar to ix and iy(check validity of this)
  print np.nanmax(bx-ix) # see variations in x and y interpolation coordinates between background and image
  print np.nanmax(by-iy)

  # Divide i by b to get t (transfer)
  quotient = iI/bI
  diff = bI-iI 
  if divideMethod == 'linear':
    tI = quotient   # Dividing makes sense for intensity
  elif divideMethod =='log':
    tI = np.log(quotient)
  elif divideMethod =='sqrt':
    tI = np.sqrt(quotient)
  elif divideMethod =='diff':
    tI = diff
  else:
    print 'Invalid division method entered, defaulting to linear'
    tI = quotient
    
  tP =  np.arctan2(np.sin(iP-bP), np.cos(iP-bP)) # Phase difference makes more sense than quotient(probably?); wrap to -pi to pi range

  #Plotting
  fig, axes = plt.subplots(3,2, sharex='col',sharey='row',figsize=(15,15))
  for i in range(3):
      for j in range(2):
	axes[i,j].set(adjustable='box-forced',aspect='equal')
  axes[0,0].set_title('Intensity (Background)')
  p00 =axes[0,0].contourf(bx,by,bI, levels, cmap='inferno')
  axes[1,0].set_title('Intensity (Image)')
  p10 = axes[1,0].contourf(ix,iy,iI, levels, cmap='inferno')
  axes[2,0].set_title('Intensity (Quotient)')
  p20 = axes[2,0].contourf(ix,iy,tI, levels, cmap='inferno') # For now use image coordinates, fix later if image and background coordinates are too inconsistent 

  divider = make_axes_locatable(axes[0,0]) # Ugly block to make colorbars work
  cax = divider.append_axes('right',size='5%',pad=.05)
  fig.colorbar(p00,cax=cax,orientation='vertical')
  divider = make_axes_locatable(axes[1,0])
  cax = divider.append_axes('right',size='5%',pad=.05)
  fig.colorbar(p10,cax=cax,orientation='vertical')
  divider = make_axes_locatable(axes[2,0])
  cax = divider.append_axes('right',size='5%',pad=.05)
  fig.colorbar(p20,cax=cax,orientation='vertical')

  axes[0,1].set_title('Phase (Background)')
  p01 =axes[0,1].contourf(bx,by,bP, levels, cmap='hsv')
  axes[1,1].set_title('Phase (Image)')
  p11 = axes[1,1].contourf(ix,iy,iP, levels, cmap='hsv')
  axes[2,1].set_title('Phase (Difference)')
  p21 = axes[2,1].contourf(ix,iy,tP, levels, cmap='hsv') 

  divider = make_axes_locatable(axes[0,1]) # Ugly block to make colorbars work
  cax = divider.append_axes('right',size='5%',pad=.05)
  fig.colorbar(p01,cax=cax,orientation='vertical')
  divider = make_axes_locatable(axes[1,1])
  cax = divider.append_axes('right',size='5%',pad=.05)
  fig.colorbar(p11,cax=cax,orientation='vertical')
  divider = make_axes_locatable(axes[2,1])
  cax = divider.append_axes('right',size='5%',pad=.05)
  fig.colorbar(p21,cax=cax,orientation='vertical')

  axes[0,0].invert_yaxis() # Invert y-axes since acam uses positive y-axis downward
  axes[1,0].invert_yaxis()
  axes[2,0].invert_yaxis()
  plt.show()