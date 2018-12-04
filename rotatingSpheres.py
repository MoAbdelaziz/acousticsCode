import numpy as np

######################
###   Definitions  ###
######################

# Physical definitions
eta  = 1e-6  # Viscosity, Pa*s
rho  = 1e3   # Density of water, mks
kT   = 4e-21 # Temperature in J
a    = 1e-6  # All same size, meters 


# Code unit scales
A0 = a              # m, particle size scale
T0 = rho/eta * a**2 # s, hydrodynamic timescale
M0 = a**3 * rho     # kg, particle mass scale

# Code parameters
N    = 2     # Number of particles
Nt   = 8000  # Number of timesteps
dt   = T0  # Duration of one step
rCut = 10*A0  # Cutoff distance for interactions

#################
### Functions ###
#################

# Initialize particles in a box of size 2*rCut
def initialize(N,rCut):
	x = np.linspace(-rCut,rCut,N)
	gx,gy,gz = np.meshgrid(x,x,x)
	pos = np.zeros((3,N,Nt))
	return
	
# Calculate interparticle distances accounting for periodic boundaries
def separations():
	return

# Calculate Gor'kov acoustic trapping force
def gorkov():
	return

# Calculate primary Bjerknes forces
def Bjerknes1():
	return
	
# Secondary Bjerknes forces
def Bjerknes2():
	return

# Low (nonzero) Re flows from constant torque
def rotFlow():
	return

# Repulsion at small distances
def repel():
	return

# Calculate translational mobility matrices
def	mobility():
	return
		
#################
### Main Loop ###
#################
initialize(3,1)
# Calculate velocities (superpose torque-caused flows with calculation of translation mobility matrices on forces)

# Update positions

# Deal with particles intersecting? (or just use a smaller timestep?)

#############
### Plots ###
#############
