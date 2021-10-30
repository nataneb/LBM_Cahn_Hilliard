#!/usr/bin/python
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.
#
#This program is the realization of the Layered Poiseuille flow test case via the LBM
#for Cahn-Hilliard equation provided by  Zheng et al. (2015)
#
# Layered Poiseuille flow
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
from auxiliary.VTKWrapper import saveToVTK
from auxiliary.collide import BGKCollide
from auxiliary.stream import stream,stream2
from auxiliary.LBMHelpers_CHE import calculateChemicalPotential, equilibriumOrderParameter, equilibriumG, getMacroValuesFromh, getMacroValuesFromg, getInterfaceForce, getGammas,getForceTermOrderParameter, getForceTermFlowField, noslip
from auxiliary.boundaryConditions import NeutralWetting

import os
import sys


###### Plot settings ############################################################

plotEveryN    = 1000000                 # draw every plotEveryN'th cycle
skipFirstN    = 0                       # do not process the first skipFirstN cycles for plotting
savePlot      = True                    # saving corresponding plots when savePlot is True
liveUpdate    = True                    # show the process of the simulation (slow)
saveVTK       = False                   # save the vtk files
prefix1        = 'initial'              # naming prefix for saved files
prefix2        = 'rho_order_par'        # naming prefix for saved files
prefix3        = 'pres_vel'             # naming prefix for saved files
outputFolder  = './out_poiseuille_flow_nu_r_1000'   #Folder where all results are saved
workingFolder = os.getcwd()

###### Flow definition #############
maxIterations = 100000000   # Total number of iterations in time
Re            = 0.002       # Reynolds number
dt            = 1.          # Time step size in lattice units

# Number of Cells
ny = 101        #number of cells in y direction
nx = 11         #number of cells in x direction
half_ny = int(ny/2)

# Highest index in each direction, indexing starts from 0 in python
nxl = nx-1
nyl = ny-1

# number of microscopic velocities
q  = 9

cs2=1./3  #speed of sound


gavr    = 0         #gravitational force
fg1 = zeros(shape=(2,nx,ny))
fg1[1,:,:] = gavr


 
#viscosities for different layers
nu1 = 0.1
nu_r = 1000
nu2 = nu1/nu_r

#external force
a_x  = (2*nu1*Re)/((ny**2)/4)
fg1[0,:,:] = fg1[0,:,:] + a_x 

#densities
rho1 = 1
rho2 = 1

#interfacial thickness, surface tension and parameters defined by them
D=4
sigma =0.0001
kappa = 1.5*sigma*D
beta =  (12.*sigma)/D
 
gamma =1    #adjusting parameter for mobility
 
 
#order parameter maximum and minimum values
c1 = 1
c2 = 0


###### Plot preparations ##########

# quick and dirty way to create output directory
if not os.path.isdir(outputFolder):
    try:
        os.makedirs(outputFolder)
    except OSError:
        pass

###### Setup ########
#Boundaries and corresponding indices left without the values for dstribution function after streaming
boundary_lower = fromfunction(lambda x, y: (y == 0), (nx, ny))
boundary_upper = fromfunction(lambda x, y: (y == nyl), (nx, ny))
indices_lower = array([2,5,6])
indices_upper = array([4,7,8])


#Initial condition for distribution functions, velocityy and pressure
c = ones(shape=(nx, ny))
rho = zeros(shape=(nx, ny))+rho1
ueq = zeros(shape=(2,nx,ny))
pressin = zeros(shape=(nx,ny))

#Exact solution for 1st component of velocity
uex_x = fromfunction(lambda x, y: (a_x*half_ny**2)/(2*((y<half_ny)*nu1+(y>=half_ny)*nu2))*(-((y-half_ny)/half_ny)**2+((nu1-nu2)/(nu1+nu2))*((y-half_ny)/half_ny)+2*((y<half_ny)*nu1+(y>=half_ny)*nu2)/(nu1+nu2)), (1,ny))


uex_x = uex_x[0,:]

#mobility (better to be here since it can sometimes be dependent on order paramter)
M = 0.0
 
# Relaxation parameters
tau_h = M/(dt*gamma) 
omega_h = 2*dt/(2*tau_h+dt)
tau_f_1 = 3*nu1/dt+0.5
tau_f_2 = 3*nu2/dt+0.5
omega_f = zeros((nx, ny)) + (1./tau_f_1)
omega_f[:,int(nyl/2):ny+1] = (1./tau_f_2)

print("tau_h\t", tau_h, "\tomega_h\t", omega_h, "\n")
print("omega1\t", 1./tau_f_1)
print("omega2\t", 1./tau_f_2)

#Chemical potential
mu = calculateChemicalPotential(c, beta, c1, c2, kappa, dt, 'classical', 'neutral wetting', True, True)
#Interface force
f_s = getInterfaceForce(mu, c, dt, 'neutral wetting', True, True)
#Adding body force
F = fg1 + f_s 

#Equilibiurm distribution functions
heq=equilibriumOrderParameter(gamma,ueq, mu, c)

geq = equilibriumG(rho,ueq,pressin, dt)

#Initialization of post Streaming and post Collision distribution functions
hin = heq.copy()
gin = geq.copy()
hpost = heq.copy()
gpost = geq.copy()
p = pressin.copy()


os.chdir(outputFolder)

# Plot initial condition
fig, ax = pyplot.subplots(1,2)

print("rho\t", rho.shape)
ax[0].imshow(rho.transpose(), cmap=cm.Blues, interpolation='none',origin='lower')
ax[0].set_title('Initial rho', fontsize=12)
ax[1].imshow(c.transpose(), cmap=cm.Greens, interpolation='none',origin='lower')
ax[1].set_title('Initial order parameter', fontsize=12)
pyplot.tight_layout()
pyplot.savefig(prefix1 +"_"+prefix2 +".png", dpi=200)
pyplot.close()

fig, ax = pyplot.subplots(1,2)
ax[0].imshow(p.transpose(), cmap=cm.Blues, interpolation='none',origin='lower')
ax[0].set_title('Initial pressure', fontsize=12)
ax[1].imshow(sqrt(ueq[0] ** 2 + ueq[1] ** 2).transpose(), cmap=cm.Blues, interpolation='none',origin='lower')
ax[1].set_title('Initial velocity', fontsize=12)
pyplot.tight_layout()
pyplot.savefig(prefix1 +"_"+prefix3 +".png", dpi=200)
pyplot.close()

u_old  = ueq.copy()

###### Main time loop ########
for time in range(maxIterations):
    
    # Streaming step for order parameter and flow field distribution functions
    hin = stream2(hpost)
    gin = stream2(gpost)
    
    # Bounce back for upper and lower walls
    for i in indices_lower:
        gin[i, boundary_lower] = gpost[noslip[i], boundary_lower]
    for i in indices_upper:
        gin[i, boundary_upper] = gpost[noslip[i], boundary_upper]
    
    #Neutral wetting BC for upper and lower walls
    NeutralWetting(hin,True, True)
    
    #Order parameter and density
    c = getMacroValuesFromh(hin)
    rho = c*rho1+(1.-c)*rho2

    # Calculate chemical potential
    mu = calculateChemicalPotential(c, beta, c1, c2, kappa, dt, 'classical', 'neutral wetting', True, True)
    
    #Interface force
    f_s = getInterfaceForce(mu, c, dt, 'neutral wetting', True, True)
    # Adding body force
    F = fg1 + f_s 
    
    #Pressure abd velocity
    (p,u) = getMacroValuesFromg(gin, dt, rho, F,'neutral wetting', True, True)
    
    #Equilibrium distribution functions for order parameter and flow field
    geq = equilibriumG(rho,u,p, dt)
    heq = equilibriumOrderParameter(gamma, u, mu, c)
    
    #Checking extreme out-of-bound values
    if amax(c)>10 or amin(c)<-10:
        print(time,"\tmax c\t", amax(c), "\tmin c\t", amin(c))
        sys.exit()

    # Collision step
    gpost = BGKCollide(gin, geq, omega_f)
    hpost = BGKCollide(hin, heq, omega_h)
    
    #Incorporating force terms
    gpost = gpost + getForceTermFlowField(F, u, rho, dt, omega_f, 'neutral wetting', True, True)
    hpost = hpost + getForceTermOrderParameter(u, c, p, F, rho, dt, omega_h, 'neutral wetting', 'bounce back', True, True)
    
    #Checking of the fulfillment of stopping criterion and result is wrtten into file
    if time>0 and time%(int(nu_r/10)) == 0:
        u_diff = u-u_old
        diff_norm = sum(u_diff[0,:,:]**2+u_diff[1,:,:]**2, axis = (0,1,))
        u_norm = sum(u[0,:,:]**2+u[1,:,:]**2, axis = (0,1,))
        if sqrt(diff_norm/u_norm) < 1e-6:
            print("stopped after ", time+1, " time steps")
            print("Error norm ", sqrt(sum(abs(uex_x-u[0,5,:]))/sum(abs(uex_x))))
            
            file1 = open("results_Error.txt", "w")
            file1.write("Error norm "+ str(sqrt(sum((uex_x-u[0,5,:])**2)/sum(uex_x**2))))
            file1.close()
            break
        u_old = u.copy()

    # Visualization
    if ( ((time+1) % plotEveryN == 0) & (liveUpdate | saveVTK | savePlot) & (time > skipFirstN) ):
        if ( liveUpdate | savePlot ):

            gs = gridspec.GridSpec(1, 4)
            axrho = pyplot.subplot(gs[0, :2])
            axc = pyplot.subplot(gs[0, 2:])
            pyplot.tight_layout()
            axrho.clear()
            im1=axrho.imshow(rho.transpose(), cmap=cm.GnBu, interpolation='none',origin='lower')  # vmin=0., vmax=0.1)
            axrho.set_title('Density')
            pyplot.colorbar(im1, ax=axrho,orientation='horizontal')
            
            axc.clear()
            im0=axc.imshow(c.transpose(), cmap=cm.YlGn, interpolation='none',origin='lower')  # vmin=0., vmax=0.1)
            axc.set_title('Order Parameter')
            pyplot.colorbar(im0, ax=axc,orientation='horizontal')
            pyplot.savefig(prefix2 + "." + str(time+1).zfill(4) + ".png", dpi=200)
            pyplot.close()    
            
            t = arange(-50, 51, 0.01)
            s = (a_x*half_ny**2)/(2*nu1)*(-(t/half_ny)**2+(nu1-nu2)/(nu1+nu2)*(t/half_ny)+2*nu1/(nu1+nu2))
            s[t>=0] = (a_x*half_ny**2)/(2*nu2)*(-(t[t>=0]/half_ny)**2+(nu1-nu2)/(nu1+nu2)*(t[t>=0]/half_ny)+2*nu2/(nu1+nu2))
            pyplot.plot(t,s,'k')
            t = arange(-50, 51, 1)
            pyplot.plot(t,u[0,7,:], 'bx')
            pyplot.xlim(-51,51)
            pyplot.legend(['Analytic', 'LBE'])
            pyplot.savefig("result_nu_r = "+str(nu_r)+"_time ="+str(time+1)+" .png", dpi=200)
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

#Drawing numerical and analytical results in one figure
t = arange(-50, 51, 0.01)
s = (a_x*half_ny**2)/(2*nu1)*(-(t/half_ny)**2+(nu1-nu2)/(nu1+nu2)*(t/half_ny)+2*nu1/(nu1+nu2))
s[t>=0] = (a_x*half_ny**2)/(2*nu2)*(-(t[t>=0]/half_ny)**2+(nu1-nu2)/(nu1+nu2)*(t[t>=0]/half_ny)+2*nu2/(nu1+nu2))
pyplot.plot(t,s,'k')
t = arange(-50, 51, 1)
pyplot.plot(t,u[0,5,:], 'bx')
pyplot.xlim(-51,51)
pyplot.legend(['Analytic', 'LBE'])
pyplot.savefig("result_nu_r = "+str(nu_r)+".png", dpi=200)
pyplot.close()
os.chdir(workingFolder)
