#!/usr/bin/python
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.
#
#This program is the realization of energy fitting curves for the spinodal decomposition test case via the LBM
#for time-fractional Cahn-Hilliard equation provided by  Liang et al. (2020)
#
# Energy fitting (spinodal decomposition)
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
from scipy.optimize import curve_fit

import os
import sys

#Objective functions for fitting
def objective(x, a, b):
	return b*(x**a)

def objective_log(x, a, b):
	return a*x+b

###### Plot settings ###########

plotEveryN    = 1000                # draw every plotEveryN'th cycle
skipFirstN    = 0                   # do not process the first skipFirstN cycles for plotting
savePlot      = True                # saving corresponding plots when savePlot is True
liveUpdate    = True                # show the process of the simulation (slow)
saveVTK       = False               # save the vtk files
prefix1        = 'initial'          # naming prefix for saved files
prefix2        = 'order_par'        # naming prefix for saved files
prefix3        = 'energy'           # naming prefix for saved files

outputFolder  = './out_spinodal_decomposition_energies_fitting_phi_a_neg0.3'    #Folder where all results are saved
workingFolder = os.getcwd()

###### Setup of the experiment ########
maxIterations = 20000    # Total number of time iterations.
dt            = 1.       # Time step in lattice units

# Number of Cells
ny = 100        #number of cells in y direction
nx = 100        #number of cells in x direction


# Highest index in each direction, indexing starts from 0 in python
nxl = nx-1
nyl = ny-1

# number of microscopic velocities
q  = 9

cs2=1./3  #speed of sound

#interfacial thickness, surface tension and the parameters defined by them
D=3
sigma =0.1
kappa = (3./8.)*sigma*D
beta =  (3./4.)*sigma/D
#fractional order
alpha= array([1.0, 0.8])
#proportion of gas and liquid
phi_a = array([-0.3])
eta = 1 #adjusting parameter for mobility
 
 
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

#Initialization
fis = zeros((maxIterations+1, nx, ny))
ueq = zeros(shape=(2,nx,ny))

energies = zeros((size(phi_a),size(alpha),maxIterations+1))

os.chdir(outputFolder)

#mobility
M = 1.0
#Random field for the fluctuation to the initial mixture
rand_distribution = (random.rand(nx, ny)*0.01-0.005)
#Loop over proportion values of gas and liquid
for a in range(size(phi_a)):
    #Initial condition for order parameter distribution function
    fi = phi_a[a]+ rand_distribution
    fis[0,:,:] = fi[:,:]
    
    #plotting initial condition
    gs = gridspec.GridSpec(1, 1)
    axfi = pyplot.subplot(gs[0, 0])
    pyplot.tight_layout()
    axfi.clear()
    im1=axfi.imshow(fis[0].transpose(), cmap=cm.jet, interpolation='none',origin='lower')
    axfi.set_title("Order parameter")
    pyplot.colorbar(im1, ax=axfi,orientation='horizontal') 
    pyplot.savefig(prefix1 + "_" + prefix2 + "_phi_a = "+str(phi_a[a])+".png", dpi=200)
    pyplot.close()
    energies[a,0,0] = getEnergy(fi, dt, fi1, fi2, beta, kappa)
    energies[a,1,0] = energies[a,0,0]
    for i in range (size(alpha)):
        gamma_coef = ((dt)**(1-alpha[i]))/(gamma(2-alpha[i]))
        
        # Relaxation parameters
        tau = M/(dt*eta*cs2)+0.5 
        omega = 1./tau

        print("tau\t", tau, "\tomega\t", omega, "\n")
        print("gamma_coef\t", gamma_coef)
        
        #Chemical potential
        mu = calculateChemicalPotential(fis[0,:,:], beta, fi1, fi2, kappa, dt, 'fractional')
        #Forcing functi
        F = getForcingFunction(fis, gamma_coef, dt, alpha[i], 0)
        #Equilibrium distribution function
        feq = equilibriumOrderParameterFractional(gamma_coef, eta, ueq, mu, fis[0,:,:])
        #Forcing distribution function
        F_is = getForcingDistributionFunctionFractional(fis[0,:,:], fis[0, :,:], F, dt, tau, ueq, cs2)

        #Initialization of post streaming and post collision distribution functions
        fin = feq.copy()
        fpost = feq.copy()
        
        ###### Main time loop ########
        for time in range(maxIterations):
                
            # Collision step
            fpost = BGKCollide(fin, feq, omega)
            fpost = fpost + F_is *dt

            #Streaming step
            fin = stream2(fpost)
            # Periodic boundary conditions (already included)
            
            #forcing function
            F = getForcingFunction(fis, gamma_coef, dt, alpha[i], time+1)
            
            #order parameter
            fis[time+1,:,:] = getMacroValuesFractional(fin, gamma_coef, F, dt) 
            #forcing distribution function
            F_is = getForcingDistributionFunctionFractional(fis[time+1,:,:], fis[time, :,:], F, dt, tau, ueq, cs2)
            
            #chemical potential
            mu = calculateChemicalPotential(fis[time+1,:,:], beta, fi1, fi2, kappa, dt, 'fractional')
            
            #equilibrium distribution function
            feq = equilibriumOrderParameterFractional(gamma_coef, eta, ueq, mu, fis[time+1,:,:])

            #Calculating energy
            energies[a,i, time+1] = getEnergy(fis[time+1,:,:], dt, fi1, fi2, beta, kappa)
            
        

# curve fit for alpha=1

fig, ax = pyplot.subplots()
x = arange(2000, maxIterations+1,1)
y = energies[0,0,2000:maxIterations+1]
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b = popt
print('y = %.5f * x ^ %.5f' % (b, a))

ax.loglog(x, y, 'k--')
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b)
# create a line plot for the mapping function
ax.loglog(x_line, y_line, 'k-')

ax.annotate("slope="+"{:.3f}".format(a), xy=(4000, 100), xytext = (5000,30),arrowprops=dict(arrowstyle="<-"))

# curve fit for alpha=0.8
x = arange(5000, maxIterations+1,1)
y = energies[0,1,5000:maxIterations+1]
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b = popt
print('y = %.5f * x ^ %.5f' % (b, a))

ax.loglog(x, y, 'k-.')
x_line = arange(min(x), max(x), 1)
# calculate the output for the range
y_line = objective(x_line, a, b)
# create a line plot for the mapping function
ax.loglog(x_line, y_line, 'b-')


ax.annotate("slope="+"{:.3f}".format(a), xy=(7000, 200), xytext = (4000,500), arrowprops=dict(arrowstyle="<-"))

#Setting axis limits
ax.set_xlim(1000,45000)
ax.set_ylim(10,1000)
ax.set_xlabel('t')
ax.set_ylabel('E')
pyplot.legend(['LB simulation, '+r'$\alpha = $'+str(alpha[0]), 'Fitting curve, '+r'$\alpha = $'+str(alpha[0]),'LB simulation, '+r'$\alpha = $'+str(alpha[1]), 'Fitting curve, '+r'$\alpha = $'+str(alpha[1])])
fig.savefig("Energies_fitting.png", dpi=200)
pyplot.close()

os.chdir(workingFolder)
