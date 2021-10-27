#!/usr/bin/python
# E-mail contact: natalia.nebulishvili@tum.de
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.
#
#This program is the realization of the test case of the rotating circular disk via the LBM
#for time-fractional Cahn-Hilliard equation provided by  Liang et al. (2020)
#
# Rotating circular disk
#
# D2Q9 Stencil with enumeration
#
# 6   2   5
#   \ | /
# 3 - 0 - 1
#   / | \
# 7   4   8

from numpy import *
import math
from scipy.special import gamma
from matplotlib import cm, pyplot, gridspec
from auxiliary.VTKWrapper import saveToVTK
from auxiliary.collide import BGKCollide
from auxiliary.stream import stream,stream2
from auxiliary.LBMHelpers_CHE import calculateChemicalPotential, equilibriumOrderParameterFractional, getMacroValuesFractional, getForcingFunction, getForcingDistributionFunctionFractional

import os
import sys


###### Plot settings ###########

plotEveryN    = 1000            # Plot and save every plotEveryNth iteration result
skipFirstN    = 0               # do not process the first skipFirstN cycles
savePlot      = True            # saving corresponding plots when savePlot is True
liveUpdate    = True            # show the process of the simulation (slow)
saveVTK       = False           # save the vtk files
prefix1        = 'disk_shape'   # naming prefix for saved files
prefix2        = 'order_par'    # naming prefix for saved files
prefix3        = 'x_axis'
outputFolder  = './out_rotating_circular_disk_fractional_CH_different_alphas' #Folder where all results are saved
workingFolder = os.getcwd()

###### Setup of the experiment #########
dt            = 1.       # time step

# Number of Cells
ny = 100        #number of cells in y direction
nx = 100        #number of cells in x direction

U0 = 0.02       #Reference velocity, used in the rotation of the disk
maxIterations = round((2*nx)/U0)    #maximum number of iterations, which is necessary for the one full rotation

# Highest index in each direction, indexing starts from 0 in python
nxl = nx-1
nyl = ny-1

# number of microscopic velocities
q  = 9

cs2=1./3  #speed of sound

#interfacial thickness, surface tension and parameters defined by them
D=4
sigma =0.1
kappa = (3./8.)*sigma*D
beta =  (3./4.)*sigma/D

#fractional order
alpha = array([1.0, 0.8,0.5])
eta = 1     #adjusting parameter for mobility
 
 
 
#order parameter maximum and minimum values
fi1 = 1
fi2 = -1



###### Plot preparations ############################################################

# quick and dirty way to create output directory
if not os.path.isdir(outputFolder):
    try:
        os.makedirs(outputFolder)
    except OSError:
        pass

#Boundary of the domain

boundary = fromfunction(lambda x, y: logical_or((y == 0), (y == ny)), (nx, ny))

# Coordinates of the bubble place in liquid with an initial smooth interface
cx = nx*0.5
cy = ny*0.5
r  = 25


#Initial condition for distribution function and velocity components

fi = fromfunction(lambda x, y: tanh(2*(r-sqrt((x-cx)**2+(y-cy)**2))/D), (nx, ny))
fis = zeros((maxIterations+1, nx, ny))
fis[0,:,:] = fi[:,:]

fis_along_x = zeros((size(alpha), nx))
ueq = zeros(shape=(2,nx,ny))

ueq[0,:,:] = fromfunction(lambda x, y: -U0*pi*(y/ny-0.5), (nx, ny))
ueq[1,:,:] = fromfunction(lambda x, y: U0*pi*(x/nx-0.5), (nx, ny))


#Mobility parameter (suggested to define here since it can sometimes depend on order paramter)
M = 0.01

os.chdir(outputFolder)

#Defining and drawing the exact shape of the disk

theta = linspace(0, 2*pi, 100)
x1 = r*cos(theta) +cx
x2 = r*sin(theta) + cy
pyplot.plot(x1,x2)
pyplot.axis("equal")

fis_shape = zeros((size(alpha),nx, ny))

#Main loop over different fractional orders
for i in range(size(alpha)):
    fis = zeros((maxIterations+1, nx, ny))
    fis[0,:,:] = fi[:,:]
    gamma_coef = ((dt)**(1-alpha[i]))/(gamma(2-alpha[i]))

    # Relaxation parameters
    tau = M/(dt*eta*cs2)+0.5 
    omega = 1./tau

    print("tau\t", tau, "\tomega\t", omega, "\n")
    print("gamma_coef\t", gamma_coef)
    
    #Chemical potential
    mu = calculateChemicalPotential(fis[0,:,:], beta, fi1, fi2, kappa, dt, 'fractional')
    #Forcing fucntion
    F = getForcingFunction(fis, gamma_coef, dt, alpha[i], 0)
    #Equilibrium distribution function for order parameter
    feq = equilibriumOrderParameterFractional(gamma_coef, eta, ueq, mu, fis[0,:,:])
    #Forcing distribution function
    F_is = getForcingDistributionFunctionFractional(fis[0,:,:], fis[0, :,:], F, dt, tau, ueq, cs2)

    #Intiializationo of post streaming and post collision order parameter distribution functions
    fin = feq.copy()
    fpost = feq.copy()


    Err1=0
    Err2=0
    #print("max fin", amax(fin))
    #print("max feq", amax(feq))
    #sourceFile = open('demo.txt', 'w')
    ###### Main time loop ##########################################################
    final_time=maxIterations
    for time in range(maxIterations):
            
        # Collision step
        fpost = BGKCollide(fin, feq, omega)
        fpost = fpost + F_is *dt
        
        #Streaming step
        fin = stream2(fpost)
        
        #Forcing function
        F = getForcingFunction(fis, gamma_coef, dt, alpha[i], time+1)
        
        #Order parameter calculation
        fis[time+1,:,:] = getMacroValuesFractional(fin, gamma_coef, F, dt) 
        
        #Foring distribution function
        F_is = getForcingDistributionFunctionFractional(fis[time+1,:,:], fis[time, :,:], F, dt, tau, ueq, cs2)
        
        #Chemical potential
        mu = calculateChemicalPotential(fis[time+1,:,:], beta, fi1, fi2, kappa, dt, 'fractional')
        
        #Equilibrium distribution function of order parameter
        feq = equilibriumOrderParameterFractional(gamma_coef, eta, ueq, mu, fis[time+1,:,:])
        
        #Maximum error between numerical and analytical solutions for defining the shape of the disk afterwards
        Err = amax(abs(fis[time+1,:,:]-fi), axis=(0,1))

        # Periodic boundary conditions (already included)
        
    #Defining the shape of the disk after simulation
    fis_shape[i,(fis[min(maxIterations, final_time),:,:]>-Err)&(fis[min(maxIterations, final_time),:,:]<Err)] = 1
    fis_along_x[i, :] = fis[min(maxIterations, final_time),:,50] 

#Drawing the shape of the disk based on the numerical results for different fractional orders    
pyplot.plot(argwhere(fis_shape[0]==1).transpose()[0],argwhere(fis_shape[0]==1).transpose()[1], 'or')
pyplot.plot(argwhere(fis_shape[1]==1).transpose()[0],argwhere(fis_shape[1]==1).transpose()[1], 'g^')
pyplot.plot(argwhere(fis_shape[2]==1).transpose()[0],argwhere(fis_shape[2]==1).transpose()[1], 'mx')
pyplot.xlim(0,100)
pyplot.ylim(0,100)
pyplot.legend(["Analytical solution", "LB simulation, "+r'$\alpha=$'+str(alpha[0]), "LB simulation, "+r'$\alpha=$'+str(alpha[1]), "LB simulation, "+r'$\alpha=$'+str(alpha[2])])
pyplot.savefig(prefix1 +".png", dpi=200)

#Drawing the numerical (for different fractional orders) and analytical solutions of order parameter along x axis
figure, axes = pyplot.subplots()
t = arange(0.0, nx, 0.01)
s = tanh(2*(r-sqrt((t-50)**2))/D)
pyplot.plot(t,s)
t = arange(0.0, 100, 1)
pyplot.plot(t,fis_along_x[0], 'ob')
pyplot.plot(t,fis_along_x[1], '^g')
pyplot.plot(t,fis_along_x[2], 'xm')
pyplot.xlabel('x')
pyplot.ylabel(r'$\phi$')
pyplot.xlim(0,100)
pyplot.ylim(-1.1,1.1)
pyplot.legend(["Analytical solution", "LB simulation, "+r'$\alpha=$'+str(alpha[0]), "LB simulation, "+r'$\alpha=$'+str(alpha[1]), "LB simulation, "+r'$\alpha=$'+str(alpha[2])])
pyplot.savefig(prefix2 + "_" + prefix3 + ".png", dpi=200)

os.chdir(workingFolder)
