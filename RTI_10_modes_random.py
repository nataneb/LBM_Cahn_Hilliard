#!/usr/bin/python
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.
#
#This program is the realization of the Rayleigh-Taylor instability test case via the LBM
#for Cahn-Hilliard equation provided by  Zheng et al. (2015)
#
# Rayleigh-Taylor instability with 10 modes
#
# D2Q9 Stencil with enumeration
#
# 6   2   5
#   \ | /
# 3 - 0 - 1
#   / | \
# 7   4   8
#


from numpy import *
from matplotlib import cm, pyplot, gridspec
from auxiliary.collide import BGKCollide
from auxiliary.stream import stream,stream2
from auxiliary.LBMHelpers_CHE import calculateChemicalPotential, equilibriumOrderParameter, equilibriumG, getMacroValuesFromh, getMacroValuesFromg, getInterfaceForce, getGammas,getForceTermOrderParameter, getForceTermFlowField, noslip
from auxiliary.boundaryConditions import NeutralWetting

import os
import sys

#Function for getting the initial perturbation with 10 modes
def getH(a_ns,b_ns, ns, x,d,w):
    sum1 = sum(a_ns[:,None,None]*cos((2*pi*ns[:, None,None])*(x[None, :,:])/w)+b_ns[:, None,None]*sin((2*pi*ns[:, None,None])*(x[None, :,:])/w), axis=0)
    
    return sum1


###### Plot settings ############################################################

plotEveryN    = 100                     # draw every plotEveryN'th cycle
skipFirstN    = 0                       # do not process the first skipFirstN cycles for plotting
savePlot      = True                    # saving corresponding plots when savePlot is True
liveUpdate    = True                    # show the process of the simulation (slow)
saveVTK       = False                   # save the vtk files
prefix1        = 'initial'              # naming prefix for saved files
prefix2        = 'rho_order_par'        # naming prefix for saved files
prefix3        = 'pres_vel'             # naming prefix for saved files
prefix4        = 'rho'                  # naming prefix for saved files
outputFolder  = './RTI_10_modes_M_0.01'
workingFolder = os.getcwd()
aveSkip       = 10000

###### Flow definition ###################################
maxIterations = 20000       # Total number of iterations in time
Re            = 8000        # Reynolds number
dt            = 1.          # Time step size in lattice units
u_r = 0.04                  # Characteristic velocity
At = 0.5                    # Atwood number

d = 256                     # Characteristic length
lambda_ = 128               # Wavelength

# Number of Cells
ny = 2*d+1       #number of cells in y direction
nx = 2*d+1       #number of cells in x direction
half_ny = int(ny/2)

# Highest index in each direction, indexing in python starts from 0
nxl = nx-1
nyl = ny-1

# number of microscopic velocities
q  = 9

cs2=1./3  #speed of sound


gavr    = (u_r**2)/d                    #Gravitational acceleration
print("g:\t", gavr)
fg1 = zeros(shape=(2,nx,ny))            
plotEveryN = int(sqrt(d/gavr))          #Characteristic time
maxIterations = int(5*sqrt(d/gavr))     #Redefinition of maximum number of iterations based on characteristic time


 #Viscosities
nu1 = (u_r*d)/Re
nu_r = 1
nu2 = nu1/nu_r


#Densities
rho1 = 1
rho2 = rho1*(1-At)/(1+At)


#Interfacial thickness, surface tension and parameters defined by them
D=5
sigma = 0
kappa = 1.5*sigma*D
beta =  (12.*sigma)/D
 
gamma =1

 
 
#order parameter maximum and minimum values
c1 = 1
c2 = 0

###### Plot preparations ########

# quick and dirty way to create output directory
if not os.path.isdir(outputFolder):
    try:
        os.makedirs(outputFolder)
    except OSError:
        pass

###### Setup #########
boundary_lower = fromfunction(lambda x, y: (y == 0), (nx, ny))
boundary_upper = fromfunction(lambda x, y: (y == nyl), (nx, ny))
indices_lower = array([2,5,6])
indices_upper = array([4,7,8])


#Coefficients for initial perturbation
a_ns = random.normal(-11,0.01, size=(10))
b_ns = random.normal(-11,0.01, size=(10))
ns = arange(1,11)

#Initial condition for distribution functions, velocity and pressure
c = fromfunction(lambda x, y: 0.5*(1+tanh(2*(y-(half_ny+getH(a_ns,b_ns,ns,x,d,2*d)))/D)), (nx,ny))
rho = c*rho1+(1-c)*rho2
ueq = zeros(shape=(2,nx,ny))
pressin = zeros(shape=(nx,ny))

#Body force (gravitational force in this case)
fg1[1,:,:] = -gavr*rho          

#mobility (better to be here since it can sometimes be dependent on order paramter)
M = 0.01
print("mobility\t", M)
 
# Relaxation parameters
tau_h = M/(dt*gamma) 
omega_h = 2*dt/(2*tau_h+dt)
tau_f_1 = 3*nu1/dt+0.5
tau_f_2 = 3*nu2/dt+0.5

print("tau_h\t", tau_h, "\tomega_h\t", omega_h, "\n")
print("omega1\t", 1./tau_f_1)
print("omega2\t", 1./tau_f_2)

#Chemical potential
mu = calculateChemicalPotential(c, beta, c1, c2, kappa, dt, 'classical', 'neutral wetting', True, True)
#Interface force
f_s = getInterfaceForce(mu, c, dt, 'neutral wetting', True, True)
#Incorporating gravitational force
F = fg1 + f_s 

#Equilibrium distribution functions for order parameter and flow field
heq=equilibriumOrderParameter(gamma,ueq, mu, c)
geq = equilibriumG(rho,ueq,pressin, dt)

#Initialization of post Streaming and post Collision distribution functions
hin = heq.copy()
gin = geq.copy()
hpost = heq.copy()
gpost = geq.copy()
p = pressin.copy()      #pressure

os.chdir(outputFolder)

# Plot initial condition
gs = gridspec.GridSpec(1, 4)
axrho = pyplot.subplot(gs[0, :2])
axc = pyplot.subplot(gs[0, 2:])
pyplot.tight_layout()
axrho.clear()
im1=axrho.imshow(rho.transpose(), cmap=cm.jet, interpolation='none',origin='lower')  # vmin=0., vmax=0.1)
axrho.set_title('Density')
pyplot.colorbar(im1, ax=axrho,orientation='horizontal')

axc.clear()
im0=axc.imshow(c.transpose(), cmap=cm.YlGn, interpolation='none',origin='lower')  # vmin=0., vmax=0.1)
axc.set_title('Order Parameter')
pyplot.colorbar(im0, ax=axc,orientation='horizontal')
pyplot.savefig(prefix2 + ".initial.png", dpi=200)
pyplot.close()    

gs = gridspec.GridSpec(1, 4)
ax1 = pyplot.subplot(gs[0, :2])
ax2 = pyplot.subplot(gs[0, 2:])
pyplot.tight_layout()

ax1.clear()
im2=ax1.imshow(sqrt(ueq[0] ** 2 + ueq   [1] ** 2).transpose(),  cmap=cm.YlOrRd,interpolation='none',origin='lower') #vmin=0., vmax=0.1)
ax1.set_title('Velocity')
pyplot.colorbar(im2, ax=ax1,orientation='horizontal')

ax2.clear()
im3=ax2.imshow(p.transpose(), cmap=cm.YlGn, interpolation='none',origin='lower')  # vmin=0., vmax=0.1)
ax2.set_title('Pressure')
pyplot.colorbar(im3, ax=ax2,orientation='horizontal')
pyplot.savefig(prefix3 + ".initial.png", dpi=200)
pyplot.close()

#plotting density only
gs = gridspec.GridSpec(1, 1)
axfi = pyplot.subplot(gs[0, 0])
pyplot.tight_layout()
axfi.clear()
im1=axfi.imshow(rho.transpose(), cmap=cm.jet, interpolation='none',origin='lower', vmin = rho2, vmax = rho1)  # vmin=0., vmax=0.1)
axfi.set_title("Density")
pyplot.savefig(prefix4 + "initial.png", dpi=200)
pyplot.close()

###### Main time loop #####
for time in range(maxIterations):
    
    # Streaming step
    hin = stream2(hpost)
    gin = stream2(gpost)
    
    # Bounce back scheme for upper and lower walls
    for i in indices_lower:
        gin[i, boundary_lower] = gpost[noslip[i], boundary_lower]
    for i in indices_upper:
        gin[i, boundary_upper] = gpost[noslip[i], boundary_upper]
    
    #Neutral wetting BC for upper and lower walls
    NeutralWetting(hin, True, True)

    #Order parameter and density
    c = getMacroValuesFromh(hin)
    rho = c*rho1+(1.-c)*rho2

 
    # Relaxation parameter for order parameter
    tau_h = M/(dt*gamma) 
    omega_h = 2*dt/(2*tau_h+dt)

    # Calculate chemical potential
    mu = calculateChemicalPotential(c, beta, c1, c2, kappa, dt, 'classical', 'neutral wetting', True, True)
    #Interface force
    f_s = getInterfaceForce(mu, c, dt, 'neutral wetting', True, True)
    #Gravitational force
    fg1[1,:,:] = -gavr*rho
    # Total force
    F = fg1 + f_s 
    
    #Pressure and velocity
    (p,u) = getMacroValuesFromg(gin, dt, rho, F,'neutral wetting', True, True)
    
    #Equilibrium distribution functions for flow field and order parameter
    geq = equilibriumG(rho,u,p, dt)
    heq = equilibriumOrderParameter(gamma, u, mu, c)

    
    #Relaxation rate for flow field
    omega_f = c*(1./tau_f_1)+(1.-c)*(1./tau_f_2)
    if amax(c)>2 or amin(c)<-1:
        print(time)
        print("max c\t", amax(c), "\tmin c\t", amin(c))
        sys.exit()

    # Collision step
    gpost = BGKCollide(gin, geq, omega_f)
    hpost = BGKCollide(hin, heq, omega_h)
    
    #Incorporating force terms
    gpost = gpost + getForceTermFlowField(F, u, rho, dt, omega_f, 'neutral wetting', True, True)
    hpost = hpost + getForceTermOrderParameter(u, c, p, F, rho, dt, omega_h, 'neutral wetting', 'bounce back', True, True)


    # Visualization
    if ( ((time+1) % plotEveryN == 0) & (liveUpdate | saveVTK | savePlot) & (time > skipFirstN) ):
        if ( liveUpdate | savePlot ):

            gs = gridspec.GridSpec(1, 4)
            axrho = pyplot.subplot(gs[0, :2])
            axc = pyplot.subplot(gs[0, 2:])
            pyplot.tight_layout()
            axrho.clear()
            im1=axrho.imshow(rho.transpose(), cmap=cm.jet, interpolation='none',origin='lower')  # vmin=0., vmax=0.1)
            axrho.set_title('Density')
            pyplot.colorbar(im1, ax=axrho,orientation='horizontal')
            
            axc.clear()
            im0=axc.imshow(c.transpose(), cmap=cm.YlGn, interpolation='none',origin='lower')  # vmin=0., vmax=0.1)
            axc.set_title('Order Parameter')
            pyplot.colorbar(im0, ax=axc,orientation='horizontal')
            pyplot.savefig(prefix2 + "." + str(time +1).zfill(4) + ".png", dpi=200)
            pyplot.close()    

            gs = gridspec.GridSpec(1, 4)
            ax1 = pyplot.subplot(gs[0, :2])
            ax2 = pyplot.subplot(gs[0, 2:])
            pyplot.tight_layout()

            ax1.clear()
            im2=ax1.imshow(sqrt(u[0] ** 2 + u[1] ** 2).transpose(),  cmap=cm.YlOrRd,interpolation='none',origin='lower') #vmin=0., vmax=0.1)
            ax1.set_title('Velocity')
            pyplot.colorbar(im2, ax=ax1,orientation='horizontal')

            ax2.clear()
            im3=ax2.imshow(p.transpose(), cmap=cm.YlGn, interpolation='none',origin='lower')  # vmin=0., vmax=0.1)
            ax2.set_title('Pressure')
            pyplot.colorbar(im3, ax=ax2,orientation='horizontal')
            pyplot.savefig(prefix3 + "." + str(time +1).zfill(4) + ".png", dpi=200)
            pyplot.close()

            #plotting only density
            gs = gridspec.GridSpec(1, 1)
            axfi = pyplot.subplot(gs[0, 0])
            pyplot.tight_layout()
            axfi.clear()
            im1=axfi.imshow(rho.transpose(), cmap=cm.jet, interpolation='none',origin='lower', vmin = rho2, vmax = rho1)  # vmin=0., vmax=0.1)
            axfi.set_title("Density")
            pyplot.savefig(prefix4 + str(time +1).zfill(4) + ".png", dpi=200)
            pyplot.close()

os.chdir(workingFolder)
