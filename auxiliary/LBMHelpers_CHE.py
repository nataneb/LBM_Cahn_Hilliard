########################################################################
#Implementation of main parts of 2 LBMS for                            #
#classical and fractional Cahn-Hilliard equations.                     #
#All the functions are based on the two papers:                        #
#one by Zheng et al.(2015) and another one by                          #
#Liang et al. (2020).                                                  #
########################################################################
# This program is free software: you can redistribute it and/or        #
# modify it under the terms of the GNU General Public License, either  #
# version 3 of the License, or (at your option) any later version.     #
########################################################################

from numpy import *



###### Lattice Constants ########
q = 9

# lattice velocities
qsi = array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])


# Lattice weights
t      = 1./36. * ones(q)
t[1:5] = 1./9.
t[0]   = 4./9.

# index array for noslip reflection
# inverts the d2q9 stencil for use in bounceback scenarios
noslip = [0, 3, 4, 1, 2, 7, 8, 5, 6]

###### Function Definitions ####################################################


# Helper function for order parameter
def sumPopulations(fin):
    return sum(fin, axis = 0)


# Equilibrium distribution function for order parameter from (Zheng et al. 2015)
def equilibriumOrderParameter(gamma, u, mu, c):
    (d, nx, ny) = u.shape
    
    qsiu   = 3.0 * dot(qsi, u.transpose(1, 0, 2))
    usqr = 3.*(u[0, :, :]**2+u[1, :, :]**2)
    
    H = zeros ((q, nx, ny))
    heq = zeros((q, nx, ny))
    
    #calculating H coefficient for using then in  heq (12) (Zheng et al. 2015)
    for i in range(q):
        H[i,:,:] = 3.*gamma*mu+c
    H[0,:,:] = (c-(1-t[0])*(3*gamma*mu+c))/t[0]
    
    #calculating equilibtium distribution function (11) (Zheng et al. 2015)
    for i in range(q):
        heq[i,:,:] = t[i]*(H[i,:,:]+c*(qsiu[i,:,:]+(qsiu[i,:,:]**2)/2.-usqr/2.))

    return heq


# Equilibrium distribution function for flow field (21) (Zheng et al. 2015)

def equilibriumG(rho, u, p, deltat):
    (d, nx, ny) = u.shape
    
    geq = zeros((q,nx,ny))
    qsiu = 3.*dot(qsi, u.transpose(1,0,2))
    u_sqr = 3.*(u[0,:,:]**2+u[1,:,:]**2)
    
    for i in range(q):
        geq[i,:,:] = t[i]*(p+(rho/3.)*(qsiu[i,:,:]+(qsiu[i,:,:]**2)/2.-u_sqr/2.))
    
    return geq

#Equilibrium distribution function for order parameter (14) (Liang et al. 2020)

def equilibriumOrderParameterFractional(gamma, eta, u, mu, fi):
    (d, nx, ny) = u.shape
    feq = zeros((q, nx, ny))
    qsiu   = 3.0 * dot(qsi, u.transpose(1, 0, 2))
    for i in range(9):
        if i==0:
            feq[i,:,:] = gamma*fi+(t[i]-1)*eta*mu
        else:
            feq[i,:,:] =   t[i]*fi*qsiu[i,:,:] + t[i]*eta*mu
            
    return feq
#Force term in collision process for order parameter distribution function (17) (Zheng et al. 2015)
#Boundary conditions are needed for calculating gradient and laplacian 
def getForceTermOrderParameter(u, c, p, F, rho, deltat, omega, BC1 = 'periodic', BC2 = 'periodic', upper = False, lower = False, left = False, right = False ):
    grad_c = getGradient(c, deltat, BC1, upper, lower, left, right)
    grad_p = getGradient(p, deltat, BC2, upper, lower, left, right)
    
    gammas = getGammas(u)
    
    (d,nx,ny) = u.shape
    (q,d) = qsi.shape
    
    qsi_u = zeros((q,d,nx,ny))
    for i in range(q):
        qsi_u[i,:,:,:] = (qsi[i] - u.transpose()).transpose()
        
    dot_prod_2nd_term = grad_c-3.*(c/rho)*(grad_p-F)
    dot_product = sum(qsi_u*dot_prod_2nd_term, axis =-3) 

    return deltat*(1-0.5*omega)*dot_product*gammas

#Force term in collision process for flow fieldr distribution function (20) (Zheng et al. 2015)
#Boundary conditions are needed for calculating gradient and laplacian 
def getForceTermFlowField(F, u, rho, deltat, omega, BC = 'periodic', upper = False, lower = False, left = False, right = False):
    grad_rho = getGradient(rho, deltat, BC, upper, lower, left, right)/3.
    gammas = getGammas(u)
    
    (d,nx,ny) = u.shape
    (q,d) = qsi.shape
    qsi_u = zeros((q,d,nx,ny))
    
    u_0 = zeros((d, nx, ny))
    gammas0 = getGammas(u_0)
    
    for i in range(q):
        qsi_u[i,:,:,:] = (qsi[i] - u.transpose()).transpose()
    qsi_ugamma = qsi_u[:,:,:,:]*gammas[:,None,:,:]
    Fqsi_uGamma = sum(F*(qsi_u[:,:,:,:]*gammas[:,None,:,:]), axis = -3)
    qsi_uGradrhoGamma = sum(qsi_u*(grad_rho[None,:,:,:]*(gammas-gammas0)[:,None,:,:]),axis=1)
    
    return deltat*(1-0.5*omega)*(Fqsi_uGamma+qsi_uGradrhoGamma)

#Forcing function (9) (Liang et al. 2020)
def getForcingFunction(fis, gamma, deltat, alpha, n):
    (n1, nx, ny) = fis.shape
    F= zeros((nx,ny))
    for i in range(1,n):
        F = F + ((n-i+1)**(1-alpha)-(n-i)**(1-alpha))*(fis[i,:,:]-fis[i-1,:,:])
    F = -(gamma/deltat)*F
    return F

#Forcing distribution function (15) (Liang et al. 2020)
def getForcingDistributionFunctionFractional(fi1, fi2, F, deltat, tau, u, cs2):
    (d, nx, ny) = u.shape
    
    fiu1 = fi1*u
    fiu2 = fi2*u
    time_der_fiu  = getTimeDerivative(fiu1, fiu2, deltat)
    qsitime_der_fiu = (dot(qsi, u.transpose(1, 0, 2)))/cs2
    
    forcing_distribution = zeros((q,nx,ny))
    for i in range(q):
        forcing_distribution[i,:,:] = (1-1/(2*tau))*t[i]*(F+qsitime_der_fiu[i,:,:])
        
    return forcing_distribution
    

#chemical potential (1) (Zheng et al. 2015) and (2) (Liang et al. 2020)
#Boundary conditions are needed for calculating laplacian
def calculateChemicalPotential(c, beta, c1, c2, kappa, deltat, type = 'classical', BC = 'periodic', upper = False, lower = False, left = False, right = False ):
    (nx, ny) = c.shape

    if type =='fractional':
        laplace_c = getLaplacianFractional(c, deltat)
    else:
        laplace_c = getLaplacian(c, deltat, BC, upper, lower, left, right)

    mu = zeros((nx, ny))
    mu = 4*beta*(c-c1)*(c-c2)*(c-(c1+c2)/2.) - kappa*laplace_c
    
    return mu


#calculating order parameter using equation (18) (Zheng et. al. 2015)
def getMacroValuesFromh(h):
    #Calculate macroscopic order parameter
    c = sumPopulations(h)
    return c
    
#calcualting macroscopic values u (velocity) and p (pressure) using equations (22) (Zheng et. al. 2015) 
#boundary conditions are used in gradient calculation
def getMacroValuesFromg(g, deltat,rho, F, BC = 'periodic', upper = False, lower = False, left = False, right = False ):
    
    grad_rho = getGradient(rho, deltat, BC, upper, lower, left, right)
    u = dot(qsi.transpose(), g.transpose(1, 0, 2))
    u = u + (deltat/6.)*F
    
    u = u/(rho/3.)
    p = sumPopulations(g)+(deltat/6.)*sum(u*grad_rho, axis=0)
    
    return (p,u)

#Interface force calculation (Zheng et al. 2015)
#boundary conditions are used in gradient calculation
def getInterfaceForce(mu, c, deltat, BC = 'periodic', upper = False, lower = False, left = False, right = False):
    grad_c =getGradient(c, deltat, BC, upper, lower, left, right)

    F_s = mu*grad_c

    return F_s

#Order parameter calculation through (17) (Liang et al. 2020)
def getMacroValuesFractional(f, gamma, F, deltat):
    
    # Calculate macroscopic order parameter
    fi = (sumPopulations(f)+(deltat/2.)*F)/gamma
    
    return fi

#calculating laplacian using (0.14) (Lee & Fischer, 2006)
#need to add boundary conditions for non-periodic ones
def getGradient(f, deltat, BC = 'periodic', upper = False , lower = False, left = False, right = False):
    
    (nx, ny) = f.shape
    (q,d) = qsi.shape
    grad = zeros((d,nx, ny))
    for i in range(1,9):
        a=roll(roll(f, -qsi[i,0],  axis=0), -qsi[i,1] ,  axis=1)
        b=roll(roll(f, qsi[i,0],  axis=0), qsi[i,1] ,  axis=1)
        if upper:
            if BC == 'neutral wetting' or BC == 'bounce back':
                #print("upper")
                if i in (2,5,6):
                    a[:,ny-1] = b[:, ny-1]
                if i in (4,7,8):
                    b[:,ny-1] = a[:, ny-1]
        if lower:
            if BC == 'neutral wetting' or BC == 'bounce back':
                #print("lower")
                if i in (4,7,8):
                    a[:,0] = b[:,0]
                if i in (2,5,6):
                    b[:,0] = a[:, 0]
        if left:
            if BC == 'neutral wetting' or BC == 'bounce back':
                if i in (3,6,7):
                    a[0,:] = b[0,:]
                if i in (1,5,8):
                    b[0,:] = a[0, :]
        if right:
            if BC == 'neutral wetting' or BC == 'bounce back':
                if i in (1,5,8):
                    a[nx-1,:] = b[nx-1,:]
                if i in (3,6,7):
                    b[nx-1,:] = a[nx-1,:]    
        qsi_grad_central = (a- b)[None,:,:]*(qsi[i])[:,None, None]
        grad = grad + t[i]*qsi_grad_central
    grad = grad*1.5/deltat
    return grad

#calculating laplacian using (0.15) (Lee & Fischer, 2006)
#need to add boundary conditions for non-periodic ones
def getLaplacian(f, deltat, BC = 'periodic', upper = False , lower = False, left = False, right = False):
    (nx, ny) = f.shape
    laplacian = zeros((nx, ny))
    for i in range(1,9):
        a=roll(roll(f, -qsi[i,0],  axis=0), -qsi[i,1] ,  axis=1)
        b=roll(roll(f, qsi[i,0],  axis=0), qsi[i,1] ,  axis=1)
        #boundary conditions
        if upper:
            if BC == 'neutral wetting' or BC =='bounce back':
                if i in (2,5,6):
                    a[:,ny-1] = b[:, ny-1]
                if i in (4,7,8):
                    b[:,ny-1] = a[:, ny-1]
        if lower:
            if BC == 'neutral wetting' or BC =='bounce back':
                if i in (4,7,8):
                    a[:,0] = b[:,0]
                if i in (2,5,6):
                    b[:,0] = a[:, 0]
        if left:
            if BC == 'neutral wetting' or BC == 'bounce back':
                if i in (3,6,7):
                    a[0,:] = b[0,:]
                if i in (1,5,8):
                    b[0,:] = a[0, :]
        if right:
            if BC == 'neutral wetting' or BC == 'bounce back':
                if i in (1,5,8):
                    a[nx-1,:] = b[nx-1,:]
                if i in (3,6,7):
                    b[nx-1,:] = a[nx-1,:]  
                    
        laplacian = laplacian + t[i]*(a-2*f+b)
    laplacian = laplacian*3./(deltat*deltat)
    return laplacian

#Calculating laplacian for LBM for fracional Cahn-Hilliard equation using (19) (Liang et al. 2020)
def getLaplacianFractional(f, deltat):
    (nx, ny) = f.shape
    laplacian = zeros((nx, ny))
    for i in range(9):
        a=roll(roll(f, -qsi[i,0],  axis=0), -qsi[i,1] ,  axis=1)
        laplacian = laplacian + 2.*t[i]*(a-f)
    laplacian = (laplacian*3.)/(deltat*deltat)
    return laplacian

#Calculating time derivative for LBM for fracional Cahn-Hilliard equation using (18) (Liang et al. 2020)
def getTimeDerivative(f1, f2, deltat):
    
    return (f1-f2)/deltat

#Calculate values of gamma function using (8) (Zheng et al. 2015) (Nothing has to do with Euler's Gamma function)
def getGammas(u):
    (d, nx, ny) = u.shape
    qsiu   = 3.0 * dot(qsi, u.transpose(1, 0, 2))
    gammas = zeros((q, nx, ny))
    usqr = (3./2.)*(u[0, :, :]**2+u[1, :, :]**2)
    for i in range(q):
        gammas[i, :, :] = t[i]*(1+qsiu[i,:,:]+0.5*(qsiu[i,:,:]**2)-usqr)
    return gammas 

#Calcualtes energy using (1) (Liang et al. 2020) and standard integration method
def getEnergy(fi,deltat, fi1, fi2, beta, kappa):
    grad_fi = getGradient(fi, deltat)
    grad_fi_sqr = (grad_fi[0, :, :]**2+grad_fi[1, :, :]**2)
    integrand = beta*(fi-fi1)**2*(fi-fi2)**2+kappa*grad_fi_sqr/2
    
    return sum(integrand, axis = (0,1))


