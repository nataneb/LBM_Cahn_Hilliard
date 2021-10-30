#!/usr/bin/python
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.
#
#This program is the realization of the test case of the coalescence of two droplets via the LBM
#for time-fractional Cahn-Hilliard equation provided by  Liang et al. (2020)
#
# Droplet Coalescence
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
import math
from scipy.special import gamma
from matplotlib import cm, pyplot, gridspec
from auxiliary.VTKWrapper import saveToVTK
from auxiliary.collide import BGKCollide
from auxiliary.stream import stream,stream2
from auxiliary.LBMHelpers_CHE import calculateChemicalPotential, equilibriumOrderParameterFractional, getMacroValuesFractional, getForcingFunction, getForcingDistributionFunctionFractional, getEnergy

import os
import sys


###### Plot settings #########

plotEveryN    = 250             # draw every plotEveryN'th cycle
skipFirstN    = 0               # do not process the first skipFirstN cycles for plotting
savePlot      = True            # saving corresponding plots when savePlot is True
liveUpdate    = True            # show the process of the simulation (slow)
saveVTK       = False           # save the vtk files
prefix1        = 'initial'      # naming prefix for saved files
prefix2        = 'order_par'    # naming prefix for saved files
prefix3        = 'energy'       # naming prefix for saved files
outputFolder  = './out_droplet_coalescence_different_alphas'    #Folder where all results are saved
workingFolder = os.getcwd()

###### Setup of the experiment ###################################
maxIterations = 20000    # Total number of iterations in time
dt            = 1.       # time step in lattice units

# Number of Cells
ny = 100        #number of cells in y direction
nx = 100        #number of cells in x direction

#maxIterations = round((2*nx)/U0)

# Highest index in each direction, indexing starts from 0 in python
nxl = nx-1
nyl = ny-1

# number of microscopic velocities
q  = 9

cs2=1./3  #speed of sound

#interfacial thickness, surface tension and parameters defined by them
D=3
sigma =0.1
kappa = (3./8.)*sigma*D
beta =  (3./4.)*sigma/D

#Fractional orders
alpha= array([1.0, 0.8, 0.5])
#Adjusting parameter for mobility
eta = 1
 
 
 
#Order parameter maximum and minimum values
fi1 = 1
fi2 = -1



###### Plot preparations #######

# quick and dirty way to create output directory
if not os.path.isdir(outputFolder):
    try:
        os.makedirs(outputFolder)
    except OSError:
        pass

#Boundary of the domain
boundary = fromfunction(lambda x, y: logical_or((y == 0), (y == ny)), (nx, ny))

#Radius of the bubbles
r=12.0

#Initial condition for distribution function and velocity
fi = fromfunction(lambda x, y: tanh(2*(r-sqrt((x-50+r-(x>50)*2*r)**2+(y-50)**2))/D), (nx, ny))
fis = zeros((maxIterations+1, nx, ny))
fis[0,:,:] = fi[:,:]
ueq = zeros(shape=(2,nx,ny))
energies = zeros((size(alpha),maxIterations+1))


#mobility (better to be here since it can sometimes depend on order paramter)
M = 1.0

#Plotting initial condition
os.chdir(outputFolder)

gs = gridspec.GridSpec(1, 1)
axfi = pyplot.subplot(gs[0, 0])
pyplot.tight_layout()
axfi.clear()
im1=axfi.imshow(fi.transpose(), cmap=cm.jet, interpolation='none',origin='lower')  # vmin=0., vmax=0.1)
axfi.set_title("Order parameter")
pyplot.colorbar(im1, ax=axfi,orientation='horizontal')  
pyplot.savefig(prefix1 + "_" + prefix2 + ".png", dpi=200)
pyplot.close()

#Array for energy values
energies[0,0] = getEnergy(fi, dt, fi1, fi2, beta, kappa)
energies[1,0] = energies[0,0]
energies[2,0] = energies[0,0]

#Loop over fractional orders
for i in range (size(alpha)):
    gamma_coef = ((dt)**(1-alpha[i]))/(gamma(2-alpha[i]))
    
    # Relaxation parameters
    tau = M/(dt*eta*cs2)+0.5 
    omega = 1./tau

    print("tau\t", tau, "\tomega\t", omega, "\n")
    print("gamma_coef\t", gamma_coef)
    
    #Chemical potential
    mu = calculateChemicalPotential(fis[0,:,:], beta, fi1, fi2, kappa, dt, 'fractional')
    #Forcing function
    F = getForcingFunction(fis, gamma_coef, dt, alpha[i], 0)
    #Equilibrium distribution function of order parameter
    feq = equilibriumOrderParameterFractional(gamma_coef, eta, ueq, mu, fis[0,:,:])
    #Forcing distribution function
    F_is = getForcingDistributionFunctionFractional(fis[0,:,:], fis[0, :,:], F, dt, tau, ueq, cs2)

    # Initialization of post Streaming and post Collision distribution functions
    fin = feq.copy()
    fpost = feq.copy()
    
    ###### Main time loop ##########################################################
    for time in range(maxIterations):
            
        # Collision step
        fpost = BGKCollide(fin, feq, omega)
        fpost = fpost + F_is *dt    #Incorporating forcing distribution function
        
        #Streaming step
        fin = stream2(fpost)
        # Periodic boundary conditions (already included)
        
        #Forcing function
        F = getForcingFunction(fis, gamma_coef, dt, alpha[i], time+1)
        
        #Order parameter
        fis[time+1,:,:] = getMacroValuesFractional(fin, gamma_coef, F, dt) 
        
        #Forcing distribution function
        F_is = getForcingDistributionFunctionFractional(fis[time+1,:,:], fis[time, :,:], F, dt, tau, ueq, cs2)
        
        #Chemical potential
        mu = calculateChemicalPotential(fis[time+1,:,:], beta, fi1, fi2, kappa, dt, 'fractional')
        
        #Equilibrium distribution function
        feq = equilibriumOrderParameterFractional(gamma_coef, eta, ueq, mu, fis[time+1,:,:])

        
        # Visualization
        if ( ((time+1) % plotEveryN == 0) & (liveUpdate | saveVTK | savePlot) & (time > skipFirstN) ):
            if ( liveUpdate | savePlot ):

                gs = gridspec.GridSpec(1, 1)
                axfi = pyplot.subplot(gs[0, 0])
                #axfioveraxis = pyplot.subplot(gs[0, 2:])
                pyplot.tight_layout()
                axfi.clear()
                im1=axfi.imshow(fis[time+1].transpose(), cmap=cm.jet, interpolation='none',origin='lower')  # vmin=0., vmax=0.1)
                axfi.set_title("Order parameter")
                pyplot.colorbar(im1, ax=axfi,orientation='horizontal')
                pyplot.savefig(prefix2 + "_"+r'$\alpha =$'+ str(alpha[i]) + "_" + str(time +1).zfill(4) + ".png", dpi=200)
                pyplot.close()   
        #Calculating energy
        energies[i, time+1] = getEnergy(fis[time+1,:,:], dt, fi1, fi2, beta, kappa)
        
#Plotting energy curves
t = arange(0.0, maxIterations+1, 1)
pyplot.plot(t, energies[0,:], 'k')
pyplot.plot(t, energies[1,:], 'r')
pyplot.plot(t, energies[2,:], 'b')
pyplot.legend([r'$\alpha = $'+str(alpha[0]), r'$\alpha = $'+str(alpha[1]), r'$\alpha = $'+str(alpha[2])])
pyplot.xlabel('t')
pyplot.ylabel('E')
pyplot.xlim(0,20000)
pyplot.ylim(10,13.5)
pyplot.savefig("Energies.png", dpi=200)

os.chdir(workingFolder)
