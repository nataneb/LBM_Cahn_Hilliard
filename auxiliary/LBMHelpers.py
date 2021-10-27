from numpy import *


###### Lattice Constants #######################################################
q = 9

# lattice velocities
c = array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])


# Lattice weights
t      = 1./36. * ones(q)
t[1:5] = 1./9.
t[0]   = 4./9.

# index array for noslip reflection
# inverts the d2q9 stencil for use in bounceback scenarios
noslip = [0, 3, 4, 1, 2, 7, 8, 5, 6]

# index arrays for different sides of the d2q9 stencil
#iLeft   = arange(q)[asarray([ci[0] <  0 for ci in c])]
#iCentV  = arange(q)[asarray([ci[0] == 0 for ci in c])]
#iRight  = arange(q)[asarray([ci[0] >  0 for ci in c])]
#iTop    = arange(q)[asarray([ci[1] >  0 for ci in c])]
#iCentH  = arange(q)[asarray([ci[1] == 0 for ci in c])]
#iBot    = arange(q)[asarray([ci[1] <  0 for ci in c])]

iLeft   = [3,6,7]
iCentV  = [0, 2,4]
iRight  = [1,8,5]
iTop    = [5,2,6]
iCentH  = [0, 1,3]
iBot    = [7,4,8]

iLeftSlip = [1, 5, 8]
iRightSlip = [3, 7, 6]
iTopSlip = [8, 4, 7]
iBotSlip = [6, 2, 5]

###### Function Definitions ####################################################


# Helper function for density computation.
def sumPopulations(fin):
    return sum(fin, axis = 0)


# Equilibrium distribution function.
def equilibrium(rho, u):
    (d, nx, ny) = u.shape
    cu   = 3.0 * dot(c, u.transpose(1, 0, 2))
    usqr = 3./2.*(u[0, :, :]**2+u[1, :, :]**2)
    feq = zeros((q, nx, ny))
    for i in range(q):
        feq[i, :, :] = rho*t[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
    return feq

# Equilibrium distribution function.
def equilibrium_complex(rho, u):
    (d, nx, ny) = u.shape
    cu   = 3.0 * dot(c, u.transpose(1, 0, 2))
    usqr = 3./2.*(u[0, :, :]**2+u[1, :, :]**2)
    feq = zeros((q, nx, ny),dtype=complex)
    for i in range(q):
        feq[i, :, :] = rho*t[i]*(1.+cu[i]+0.5*cu[i]**2-usqr)
    return feq

# Linearized equilibrium distribution function.
def equilibrium_lin(rhof, uf,rho):
    (d, nx, ny) = uf.shape
    cu   = 3.0 * dot(c, uf.transpose(1, 0, 2))
    feql = zeros((q, nx, ny),dtype=complex)
    for i in range(q):
       feql[i, :, :] = rhof * t[i]  + rho*t[i]*cu[i]
    return feql

# Linearized equilibrium distribution function with background flow
def equilibrium_linflow(rhof, uf,rho,u):
    (d, nx, ny) = uf.shape
    cu   = 3.0 * dot(c, uf.transpose(1, 0, 2))
    cu1 =  dot(c, uf.transpose(1, 0, 2))
    cu2 = dot(c, u.transpose(1, 0, 2))
    cu3 = u[0, :, :]*uf[0, :, :]+u[1, :, :]*uf[1, :, :]
    feql = zeros((q, nx, ny),dtype=complex)
    for i in range(q):
       feql[i, :, :] = rhof * t[i]  + rho*t[i]*(cu[i]+9*cu1[i]*cu2[i]-3*cu3)
    return feql


# Guo's density force discretization, f:total force
def guo(u,f, omega):
    (d, nx, ny) = u.shape
    cu   = dot(c, u.transpose(1, 0, 2))
    source = zeros((q, nx, ny))

    for i in range(q):
        fgx = (3*(c[i][0]-u[0,:,:])+9*c[i][0]*cu[i])*f[0,:,:]
        fgy = (3*(c[i][1]-u[1,:,:])+9*c[i][1]*cu[i])*f[1,:,:]
        source[i,:,:] =t[i] * (1.-0.5*omega)*(fgx+fgy)
    return source

# Equilibrium distribution function.
def equilibriumIncomp(rho, u):
    (d, nx, ny) = u.shape
    cu   = 3.0 * dot(c, u.transpose(1, 0, 2))
    usqr = 3./2.*(u[0, :, :]**2+u[1, :, :]**2)
    feq = zeros((q, nx, ny))
    for i in range(q):
        feq[i, :, :] = t[i]*(rho+cu[i]+0.5*cu[i]**2-usqr)
    return feq


# Equilibrium distribution function.
def firstOrdrEquilibrium(rho, u):
    (d, nx, ny) = u.shape
    cu   = 3.0 * dot(c, u.transpose(1, 0, 2))
    feq = zeros((q, nx, ny))
    for i in range(q):
        feq[i, :, :] = rho*t[i]*(1.+cu[i])
    return feq


# Equilibrium distribution function.
def firstOrdrEquilibriumIncomp(rho, u):
    (d, nx, ny) = u.shape
    cu   = 3.0 * dot(c, u.transpose(1, 0, 2))
    feq = zeros((q, nx, ny))
    for i in range(q):
        feq[i, :, :] = t[i]*(rho+cu[i])
    return feq


def clamp(val, minVal, maxVal):
    return maximum(minVal, minimum(val, maxVal))


def getMacroValues(f):
    # Calculate macroscopic density ...
    rho = sumPopulations(f)
    # ... and velocity
    u = dot(c.transpose(), f.transpose((1, 0, 2)))/rho
    return (rho, u)


def getMacroValuesIncomp(f):
    # Calculate macroscopic density ...
    rho = sumPopulations(f)
    # ... and velocity
    u = dot(c.transpose(), f.transpose((1, 0, 2)))
    return (rho, u)



def ZouHe(rhotw1,rhotw2,rhobw1,rhobw2,u1,u2,fin,gin) :
    # bounce back distributions at walls with non equilibrium bounce back (Zou, He). Uses information after streaming

    # Top wall

    fin[4, :, -1] = fin[2, :, -1] - (2. / 3.) * rhotw1 * u1[1, :, -1]
    fin[7, :, -1] = fin[5, :, -1] + 0.5 * (fin[1, :, -1] - fin[3, :, -1]) - (1. / 6.) * rhotw1 * u1[1, :, -1]
    fin[8, :, -1]= fin[6, :, -1] - 0.5 * (fin[1, :, -1] - fin[3, :, -1]) - (1. / 6.) * rhotw1 * u1[1, :, -1]
    gin[4, :, -1] = gin[2, :, -1] - (2. / 3.) * rhotw2 * u2[1, :, -1]
    gin[7, :, -1] = gin[5, :, -1] + 0.5 * (gin[1, :, -1] - gin[3, :, -1]) - (1. / 6.) * rhotw2 * u2[1, :, -1]
    gin[8, :, -1]= gin[6, :, -1] - 0.5 * (gin[1, :, -1] - gin[3, :, -1]) - (1. / 6.) * rhotw2 * u2[1, :, -1]

    # bottom wall

    fin[2, :, 0] = fin[4, :, 0] + (2. / 3.) * rhobw1 * u1[1, :, 0]
    fin[5, :, 0]  = fin[7, :, 0] - 0.5 * (fin[1, :, 0] - fin[3, :, 0]) + (1. / 6.) * rhobw1 * u1[1, :, 0]
    fin[6, :, 0]  = fin[8, :, 0] + 0.5 * (fin[1, :, 0] - fin[3, :, 0]) + (1. / 6.) * rhobw1 * u1[1, :, 0]
    gin[2, :, 0]  = gin[4, :, 0] + (2. / 3.) * rhobw2 * u2[1, :, 0]
    gin[5, :, 0]  = gin[7, :, 0] - 0.5 * (gin[1, :, 0] - gin[3, :, 0]) + (1. / 6.) * rhobw2* u2[1, :, 0]
    gin[6, :, 0]  = gin[8, :, 0] + 0.5 * (gin[1, :, 0] - gin[3, :, 0]) + (1. / 6.) * rhobw2 * u2[1, :, 0]

    return (fin,gin)

    #  bounce back distributions at walls with Zou He method. First calculate the densities at the wall
    #rho1tw = 1. / (1. + u1eq[1, :, -1]) * ( sumPopulations(fin[iCentH, :, -1]) + 2. * sumPopulations(fin[iTop, :, -1]))
    #rho2tw = 1. / (1. + u2eq[1, :, -1]) * (sumPopulations(gin[iCentH, :, -1]) + 2. * sumPopulations(gin[iTop, :, -1]))
    #rho1bw = 1. / (1. - u1eq[1, :, 0]) * (sumPopulations(fin[iCentH, :, 0]) + 2. * sumPopulations(fin[iBot, :, 0]))
    #rho2bw = 1. / (1. - u2eq[1, :, 0]) * (sumPopulations(gin[iCentH, :, 0]) + 2. * sumPopulations(gin[iBot, :, 0]))
    #(fin, gin) = ZouHe(rho1tw, rho2tw, rho1bw, rho2bw, u1eq, u2eq, fin, gin)
