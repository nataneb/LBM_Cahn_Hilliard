#!/usr/bin/python
# E-mail contact: natalia.nebulishvili@tum.de
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License, either
# version 3 of the License, or (at your option) any later version.
#
#This program is the realization of the stationary droplet test case via the LBM
#for Cahn-Hilliard equation provided by  Zheng et al. (2015)
#
# Stationary droplet
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
from auxiliary.LBMHelpers_CHE import calculateChemicalPotential, equilibriumOrderParameter, equilibriumG, getMacroValuesFromh, getMacroValuesFromg, getInterfaceForce, getGammas, getForceTermFlowField, getForceTermOrderParameter

import os
import sys


###### Plot settings #########

plotEveryN    = 100000              # draw every plotEveryN'th cycle
skipFirstN    = 0                   # do not process the first skipFirstN cycles for plotting
savePlot      = True                # saving corresponding plots when savePlot is True
liveUpdate    = True                # show the process of the simulation (slow)
saveVTK       = False               # save the vtk files
prefix1        = 'initial'          # naming prefix for saved files
prefix2        = 'rho_order_par'    # naming prefix for saved files
prefix3        = 'pres_vel'         # naming prefix for saved files
outputFolder  = './out_stationary_droplet_all_mobilities'   #Folder where all results are saved
workingFolder = os.getcwd()

###### Flow definition #######
maxIterations = 1000000   # Total number of iterations in time
dt            = 1.        # Time step size in lattice units

# Number of Cells
ny = 101        #number of cells in y direction
nx = 101        #number of cells in x direction

# Highest index in each direction, indexing starts from 0 in python
nxl = nx-1
nyl = ny-1

# number of microscopic velocities
q  = 9

cs2=1./3  #speed of sound

gavr    = 0     #gravitational force
fg1 = zeros(shape=(2,nx,ny))
fg1[1,:,:] = gavr
 
 #viscosities
nu1 = 0.1
nu_r = 0.01
nu2 = nu1/nu_r

#densities
rho1 = 1
rho2 = 0.01

#interfacial thickness, surface tension and parameters defined by them
D=5
sigma =0.0000001
kappa = 1.5*sigma*D
beta =  (12.*sigma)/D
 
gamma =16500        #Adjusting parameter for mobility
 
 
#order parameter maximum and minimum values
c1 = 1
c2 = 0


###### Plot preparations ###############

# quick and dirty way to create output directory
if not os.path.isdir(outputFolder):
    try:
        os.makedirs(outputFolder)
    except OSError:
        pass

###### Setup ###########

#Boundary of the domain
boundary = fromfunction(lambda x, y: logical_or((y == 0), (y == nyl)), (nx, ny))

# Coordinates of the center of the bubble and its radius
cx = nxl*0.5
cy = nyl*0.5
r  = round(nyl/4)

#Initial of distribution functions
c_ex = fromfunction(lambda x, y: 0.5*(1+tanh(2*(r-sqrt((x-cx)**2+(y-cy)**2))/D)), (nx, ny))
rho_ex = fromfunction(lambda x, y: 0.5*(rho1+rho2+(rho1-rho2)*tanh(2*(r-sqrt((x-cx)**2+(y-cy)**2))/D)), (nx, ny))

#mobility parameters (better to be here since it can sometimes be dependent on order paramter)
Ms = zeros((4,nx,ny))
Ms[0,:,:] = 0.01 
Ms[1,:,:] = 0.1 
Ms[2,:,:] = 1
Ms[3,:,:] = 0.1*sqrt((c_ex**2)*((1-c_ex)**2))  

#For saving results for different mobilities
results_c = zeros((4, nx, ny))
results_rho = zeros((4,nx,ny))
pressure_diffs = zeros(4)
spurios_velocities = zeros(4)

os.chdir(outputFolder)

#Loop over mobility parameters
for m_ind in range(4):
    #Initialization of velocity and pressure
    ueq = zeros(shape=(2,nx,ny))
    pressin = zeros(shape=(nx,ny))
    
    #Chemical potential
    mu = calculateChemicalPotential(c_ex, beta, c1, c2, kappa, dt)
    #Interface force
    f_s = getInterfaceForce(mu, c_ex, dt)
    #Incorporating body force
    F = fg1 + f_s 
    
    #Equilibrium distribution functions for order parameter and flow field
    heq=equilibriumOrderParameter(gamma,ueq, mu, c_ex)
    geq = equilibriumG(rho_ex,ueq,pressin, dt)
    
    #Initialization of post Streaming and post Collision distribution functions, pressure and velocity
    hin = heq.copy()
    gin = geq.copy()
    hpost = heq.copy()
    gpost = geq.copy()
    p = pressin.copy()
    u = ueq.copy()

    # Relaxation parameters
    tau_h = Ms[m_ind]/(dt*gamma) 
    omega_h = 2*dt/(2*tau_h+dt)
    tau_f_1 = 3*nu1/dt+0.5
    tau_f_2 = 3*nu2/dt+0.5
    
    #Plotting initial condition
    fig, ax = pyplot.subplots(1,2)

    ax[0].imshow(rho_ex.transpose(), cmap=cm.Blues, interpolation='none',origin='lower')
    ax[0].set_title('Initial rho', fontsize=12)
    ax[1].imshow(c_ex.transpose(), cmap=cm.Greens, interpolation='none',origin='lower')
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


    ###### Main time loop ########
    for time in range(maxIterations):
        
        # Streaming step
        hin = stream2(hpost)
        gin = stream2(gpost)
        
        # Periodic boundary conditions (already included)
        
        #Getting order parameter and density
        c = getMacroValuesFromh(hin)
        rho = c*rho1+(1.-c)*rho2
        
        # Calculate chemical potential
        mu = calculateChemicalPotential(c, beta, c1, c2, kappa, dt)
        #Interface force
        f_s = getInterfaceForce(mu, c, dt)
        # Incorporating body force
        F = fg1 + f_s 
        
        #Calculating pressure and velocity
        (p,u) = getMacroValuesFromg(gin, dt, rho, F)
        
        #equilibrium distribution functions for order parameter and flow field
        geq = equilibriumG(rho,u,p, dt)
        heq = equilibriumOrderParameter(gamma, u, mu, c)
        
        #Differnce in pressure
        dp = abs(p-pressin)
        pressin = p
        
        #Checking out-of-bound values
        if amax(c)>10 or amin(c)<-10:
            print(time,"\tmax c\t", amax(c), "\tmin c\t", amin(c))
            sys.exit()
        
        #Calculating relaxation rate in case of changin mobility over the domain
        if m_ind == 3:
            Ms[m_ind] = 0.1*sqrt((c**2)*((1-c)**2))
            tau_h = Ms[m_ind]/(dt*gamma) 
            omega_h =2*dt/(2*tau_h+dt)
        #Calculating relaxation rate based on the viscosity as a linear sum
        omega_f = c*(1./tau_f_1)+(1.-c)*(1./tau_f_2) 
        #Checking out-of-bound values
        if amax(omega_f)>2 or amin(omega_f)<0:
            print("max omega_f\t", amax(omega_f), "\tmin omega_f\t", amin(omega_f))

        # Collision step
        gpost = BGKCollide(gin, geq, omega_f)
        hpost = BGKCollide(hin, heq, omega_h)
        
        #Incorporating force term
        gpost = gpost + getForceTermFlowField(F, u, rho, dt, omega_f)
        hpost = hpost + getForceTermOrderParameter(u, c, p, F, rho, dt, omega_h)


        # Visualization
        if ( (time % plotEveryN == 0) & (liveUpdate | saveVTK | savePlot) & (time > skipFirstN) ):
            if ( liveUpdate | savePlot ):

                #fig2, axrho = pyplot.subplots(1)
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
                pyplot.savefig(prefix2 + ".Mobility_ind_"+str(m_ind) +"time_"+ str(time).zfill(4) + ".png", dpi=200)
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
                pyplot.savefig(prefix3  + ".Mobility_ind_"+str(m_ind) +"time_" + str(time).zfill(4) + ".png", dpi=200)
                pyplot.close()


    press_center = mean(mean(p[round(nx / 2) - 4:round(nx / 2) + 4, round(ny / 2) - 4:round(ny / 2) + 4]))
    press_far = mean(mean(p[1:4, 1:4]))
    print('The pressure difference is: ', press_center - press_far)
    pressure_diffs[m_ind] = press_center - press_far
    spurios_velocities[m_ind] = amax(sqrt(u[0,:,:]**2+u[1,:,:]**2))
    results_c[m_ind,:,:] = c
    results_rho[m_ind,:,:] = rho

#Plotting the results for different mobility parameters along the exact solution
t1 = arange(0, 51, 0.01)
t2 = arange(0,51, 1)
c_ex = 0.5*(1+tanh(2*(r-sqrt(t1**2))/D))
rho_ex = 0.5*(rho1+rho2+(rho1-rho2)*tanh(2*(r-sqrt(t1**2))/D))
pyplot.plot(t1/r, c_ex, 'k')
pyplot.plot(t2/r, results_c[0,int(cx):nx,int(cy)], 'ro',fillstyle = 'none')
pyplot.plot(t2/r, results_c[1,int(cx):nx,int(cy)], 'bx',fillstyle = 'none')
pyplot.plot(t2/r, results_c[2,int(cx):nx,int(cy)], 'gs',fillstyle = 'none')
pyplot.plot(t2/r, results_c[3,int(cx):nx,int(cy)], 'm^',fillstyle = 'none')
pyplot.legend(['Analytic','M = '+str(Ms[0,0,0]),'M = '+str(Ms[1,0,0]),'M = '+str(Ms[2,0,0]),r'$M = M_b$'])
pyplot.xlabel('x/R')
pyplot.ylabel('c')

a = pyplot.axes([0.25, 0.2, .2, .2])
pyplot.plot(t2/r, results_c[0,int(cx):nx,int(cy)], 'ro',fillstyle = 'none')
pyplot.plot(t2/r, results_c[1,int(cx):nx,int(cy)], 'bx',fillstyle = 'none')
pyplot.plot(t2/r, results_c[2,int(cx):nx,int(cy)], 'gs',fillstyle = 'none')
pyplot.plot(t2/r, results_c[3,int(cx):nx,int(cy)], 'm^',fillstyle = 'none')
pyplot.xlim(0, 0.5)
pyplot.ylim(0.999,1.01)
pyplot.xticks([0,0.5])
pyplot.yticks([1,1.005,1.01])

pyplot.savefig("order_parameter.png", dpi=200)
pyplot.close()

pyplot.plot(t1/r, rho_ex, 'k')
pyplot.plot(t2/r, results_rho[0,int(cx):nx,int(cy)], 'ro',fillstyle = 'none')
pyplot.plot(t2/r, results_rho[1,int(cx):nx,int(cy)], 'bx',fillstyle = 'none')
pyplot.plot(t2/r, results_rho[2,int(cx):nx,int(cy)], 'gs',fillstyle = 'none')
pyplot.plot(t2/r, results_rho[3,int(cx):nx,int(cy)], 'm^',fillstyle = 'none')
pyplot.legend(['Analytic','M = '+str(Ms[0,0,0]),'M = '+str(Ms[1,0,0]),'M = '+str(Ms[0,0,0]),r'$M = M_b$'])
pyplot.xlabel('x/R')
pyplot.ylabel(r'$\rho$')

a = pyplot.axes([0.25, 0.2, .2, .2])
pyplot.plot(t2/r, results_rho[0,int(cx):nx,int(cy)], 'ro',fillstyle = 'none')
pyplot.plot(t2/r, results_rho[1,int(cx):nx,int(cy)], 'bx',fillstyle = 'none')
pyplot.plot(t2/r, results_rho[2,int(cx):nx,int(cy)], 'gs',fillstyle = 'none')
pyplot.plot(t2/r, results_rho[3,int(cx):nx,int(cy)], 'm^',fillstyle = 'none')
pyplot.xlim(0, 0.5)
pyplot.ylim(0.999,1.01)
pyplot.xticks([0,0.5])
pyplot.yticks([1,1.005,1.01])

pyplot.savefig("density.png", dpi=200)
pyplot.close()

#Saving maximum spurious velocities and ratios between the provided surface tension and the one calculated after simulation
file1 = open("results_velocities_surface_tension.txt", "w")
file1.write("Spurious velocities:\n"+"M = "+str(Ms[0,0,0])+"\t"+str(spurios_velocities[0])+"\n")
file1.write("M = "+str(Ms[1,0,0])+"\t"+str(spurios_velocities[1])+"\n")
file1.write("M = "+str(Ms[2,0,0])+"\t"+str(spurios_velocities[2])+"\n")
file1.write("M =Mb\t"+str(spurios_velocities[3])+"\n")

file1.write("Ratios between sigma and sigma_L:\n"+"M = "+str(Ms[0,0,0])+"\t"+str(sigma/(pressure_diffs[0]*r))+"\n")
file1.write("M = "+str(Ms[1,0,0])+"\t"+str(sigma/(pressure_diffs[1]*r))+"\n")
file1.write("M = "+str(Ms[2,0,0])+"\t"+str(sigma/(pressure_diffs[2]*r))+"\n")
file1.write("M =Mb\t"+str(sigma/(pressure_diffs[3]*r))+"\n")

file1.close()
os.chdir(workingFolder)
