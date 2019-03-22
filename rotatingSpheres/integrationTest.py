# Making sure I understand how to use the scipy integration functions
from __future__ import division

import numpy as np
import pylab as plt
import scipy.special   as sp 
import scipy.integrate as si
xList = np.linspace(0,1)

def y(x):
	return n*x**2
	
def integ(n):
	return si.simps(y(xList),xList)

for n in range(0,2):
	print 0
print n
print integ(1)
