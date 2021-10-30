#!/usr/bin/python
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.
#
#This program is the realization of the convergence test based on the test case of the stationary circular disk via the LBM
#for time-fractional Cahn-Hilliard equation provided by  Liang et al. (2020)
#
# Convergence test (static circular disk)
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
from auxiliary.LBMHelpers_CHE import calculateChemicalPotential, equilibriumOrderParameterFractional, getMacroValuesFractional, getForcingFunction, getForcingDistributionFunctionFractional

import os
import sys


###### Plot settings ############################################################

plotEveryN    = 1000            # Plot and save every plotEveryNth iteration result
skipFirstN    = 0               # do not process the first skipFirstN cycles
savePlot      = True            # saving corresponding plots when savePlot is True
liveUpdate    = True            # show the process of the simulation (slow)
saveVTK       = False           # save the vtk files
prefix1        = 'disk_shape'   # naming prefix for saved files
prefix2        = 'order_par'    # naming prefix for saved files
prefix3        = 'x_axis'
outputFolder  = './out_circular_disk_fractional_CH_convergence'     #Folder where all results are saved
workingFolder = os.getcwd()

###### Setup of the experiment ###################################
dt            = 1.       # time step in lattice units

# Number of Cells
ny = 100        #number of cells in y direction
nx = 100        #number of cells in x direction

maxIterations = 10000   #maximum number of iterations

# Highest index in each direction, indexing starts from 0 in python
nxl = nx-1
nyl = ny-1

# number of microscopic velocitie
q  = 9

cs2=1./3  #speed of sound

#Constant Cahn number and varying length of the domain
Cn = 0.04
Ls = array ([50,75,100,125,150,175,200])


#fractional order
alpha = array([1,0.8,0.5])
eta = 1

#Defining array for saving relative errors
Errors = zeros((size(Ls), size(alpha)))


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

os.chdir(outputFolder)

#Loop over domain length
for L_ind in range(size(Ls)):
    
    #interfacial thickness
    D=Cn*Ls[L_ind]
    #Number of cells in x and y direction
    nx = Ls[L_ind]
    ny = Ls[L_ind]
    #Surface tention and parameters based on it and interface thickness
    sigma =0.1
    kappa = (3./8.)*sigma*D
    beta =  (3./4.)*sigma/D
    

    #Boudary of the domain
    boundary = fromfunction(lambda x, y: logical_or((y == 0), (y == ny)), (nx, ny))

    # Center and radius of the circular disk
    cx = nx*0.5
    cy = ny*0.5
    r  = nx/4.


    #Initial condition for distribution function
    fi = fromfunction(lambda x, y: tanh(2*(r-sqrt((x-cx)**2+(y-cy)**2))/D), (nx, ny))
    fis = zeros((maxIterations+1, nx, ny))
    fis[0,:,:] = fi[:,:]
    
    #Order parameter along x axis
    fis_along_x = zeros((size(alpha), nx))
    ueq = zeros(shape=(2,nx,ny))


    #mobility (better to be here since it can sometimes depend on order paramter)
    M = 0.01

    fis_shape = zeros((size(alpha),nx, ny))
    
    #Loop over fractional orders
    for i in range(size(alpha)):
        if alpha[i] == 0.5 and Ls[L_ind ] ==200:
            continue
        #Initialization of distribution function
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
        #Forcing function
        F = getForcingFunction(fis, gamma_coef, dt, alpha[i], 0)
        #Equilibrium distribution function for order parameter
        feq = equilibriumOrderParameterFractional(gamma_coef, eta, ueq, mu, fis[0,:,:])
        #Forcing distribution function
        F_is = getForcingDistributionFunctionFractional(fis[0,:,:], fis[0, :,:], F, dt, tau, ueq, cs2)

        #Initialization of post Streaming and post Collision distribution functions
        fin = feq.copy()
        fpost = feq.copy()

        ###### Main time loop #######
        for time in range(maxIterations):
                
            # Collision step
            fpost = BGKCollide(fin, feq, omega)
            fpost = fpost + F_is *dt

            #Streaming step
            fin = stream2(fpost)            
            # Periodic boundary conditions (already included)
            
            #Forcing function
            F = getForcingFunction(fis, gamma_coef, dt, alpha[i], time+1)
            #Calculating order parameter
            fis[time+1,:,:] = getMacroValuesFractional(fin, gamma_coef, F, dt) 
            #forcing distribution function
            F_is = getForcingDistributionFunctionFractional(fis[time+1,:,:], fis[time, :,:], F, dt, tau, ueq, cs2)
            #Chemical potential
            mu = calculateChemicalPotential(fis[time+1,:,:], beta, fi1, fi2, kappa, dt, 'fractional')
            
            #Equilibrium distribution function
            feq = equilibriumOrderParameterFractional(gamma_coef, eta, ueq, mu, fis[time+1,:,:])
            
            
        #Getting order parameter along x axis
        fis_along_x[i, :] = fis[maxIterations,:,int(cy)] 
        #Calculating relative L_1 norm
        Err = sum(abs(fis[maxIterations,:,:]-fi), axis=(0,1))/sum(abs(fi), axis=(0,1))
        Errors[L_ind,i] = Err
        
        maxIterations = 10000
        
#Plotting relative L_1 norms and the line with slope -2 
s = arange(50,200,1)
t = (1/s)**2*100
pyplot.loglog(Ls, Errors[:,0], 'or')
pyplot.loglog(Ls, Errors[:,1], 'g^')
pyplot.loglog(Ls, Errors[:,2], 'mx')
pyplot.loglog(s,t,'k')
pyplot.legend(["LB simulation, "+r'$\alpha=$'+str(alpha[0]), "LB simulation, "+r'$\alpha=$'+str(alpha[1]), "LB simulation, "+r'$\alpha=$'+str(alpha[2]), "Line, slope=-2.0"])
pyplot.ylim(5e-6,0.1)
pyplot.xlim (45,240)
pyplot.xticks(Ls, Ls)
pyplot.savefig("relative_errors_10000.png", dpi=200)

os.chdir(workingFolder)
