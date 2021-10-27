from numpy import *

def centralMomentFromDistribution_Long(u, fin):
    ux = u[0, :, :]
    uy = u[1, :, :]

    fin_00 = fin[0, :, :]
    fin_10 = fin[1, :, :]
    fin_01 = fin[2, :, :]
    fin_n10 = fin[3, :, :]
    fin_0n1 = fin[4, :, :]
    fin_11 = fin[5, :, :]
    fin_n11 = fin[6, :, :]
    fin_n1n1 = fin[7, :, :]
    fin_1n1 = fin[8, :, :]

    c_00 = \
    fin_n1n1 + \
    fin_n10 + \
    fin_n11 + \
    fin_0n1 + \
    fin_00 + \
    fin_01 + \
    fin_1n1 + \
    fin_10 + \
    fin_11

    c_10 = \
    (-1 - ux) * fin_n1n1 + \
    (-1 - ux) * fin_n10 + \
    (-1 - ux) * fin_n11 + \
    (- ux)    * fin_0n1 + \
    (- ux)    * fin_00 + \
    (- ux)    * fin_01 + \
    (1 - ux)  * fin_1n1 + \
    (1 - ux)  * fin_10 + \
    (1 - ux)  * fin_11

    c_01 = \
    (-1 - uy) * fin_n1n1 + \
    ( - uy)   * fin_n10 + \
    (1 - uy)  * fin_n11 + \
    (-1 - uy) * fin_0n1 + \
    ( - uy)   * fin_00 + \
    (1 - uy)  * fin_01 + \
    (-1 - uy) * fin_1n1 + \
    ( - uy)   * fin_10 + \
    (1 - uy)  * fin_11

    c_11 = \
    (-1 - ux) * (-1 - uy) * fin_n1n1 + \
    (-1 - ux) * ( - uy)   * fin_n10 + \
    (-1 - ux) * (1 - uy)  * fin_n11 + \
    (- ux)    * (-1 - uy) * fin_0n1 + \
    (- ux)    * ( - uy)   * fin_00 + \
    (- ux)    * (1 - uy)  * fin_01 + \
    (1 - ux)  * (-1 - uy) * fin_1n1 + \
    (1 - ux)  * ( - uy)   * fin_10 + \
    (1 - ux)  * (1 - uy)  * fin_11

    c_20 = \
    square((-1 - ux)) * fin_n1n1 + \
    square((-1 - ux)) * fin_n10 + \
    square((-1 - ux)) * fin_n11 + \
    square((- ux))    * fin_0n1 + \
    square((- ux))    * fin_00 + \
    square((- ux))    * fin_01 + \
    square((1 - ux))  * fin_1n1 + \
    square((1 - ux))  * fin_10 + \
    square((1 - ux))  * fin_11

    c_02 = \
    square((-1 - uy)) * fin_n1n1 + \
    square(( - uy))   * fin_n10 + \
    square((1 - uy))  * fin_n11 + \
    square((-1 - uy)) * fin_0n1 + \
    square(( - uy))   * fin_00 + \
    square((1 - uy))  * fin_01 + \
    square((-1 - uy)) * fin_1n1 + \
    square(( - uy))   * fin_10 + \
    square((1 - uy))  * fin_11

    c_21 = \
    square((-1 - ux)) * (-1 - uy) * fin_n1n1 + \
    square((-1 - ux)) * ( - uy)   * fin_n10 + \
    square((-1 - ux)) * (1 - uy)  * fin_n11 + \
    square((- ux))    * (-1 - uy) * fin_0n1 + \
    square((- ux))    * ( - uy)   * fin_00 + \
    square((- ux))    * (1 - uy)  * fin_01 + \
    square((1 - ux))  * (-1 - uy) * fin_1n1 + \
    square((1 - ux))  * ( - uy)   * fin_10 + \
    square((1 - ux))  * (1 - uy)  * fin_11

    c_12 = \
    (-1 - ux) * square((-1 - uy)) * fin_n1n1 + \
    (-1 - ux) * square(( - uy))   * fin_n10 + \
    (-1 - ux) * square((1 - uy))  * fin_n11 + \
    (- ux)    * square((-1 - uy)) * fin_0n1 + \
    (- ux)    * square(( - uy))   * fin_00 + \
    (- ux)    * square((1 - uy))  * fin_01 + \
    (1 - ux)  * square((-1 - uy)) * fin_1n1 + \
    (1 - ux)  * square(( - uy))   * fin_10 + \
    (1 - ux)  * square((1 - uy))  * fin_11

    c_22 = \
    square((-1 - ux)) * square((-1 - uy)) * fin_n1n1 + \
    square((-1 - ux)) * square(( - uy))   * fin_n10 + \
    square((-1 - ux)) * square((1 - uy))  * fin_n11 + \
    square((- ux))    * square((-1 - uy)) * fin_0n1 + \
    square((- ux))    * square(( - uy))   * fin_00 + \
    square((- ux))    * square((1 - uy))  * fin_01 + \
    square((1 - ux))  * square((-1 - uy)) * fin_1n1 + \
    square((1 - ux))  * square(( - uy))   * fin_10 + \
    square((1 - ux))  * square((1 - uy))  * fin_11

    return (c_00, c_10, c_01, c_11, c_20, c_02, c_21, c_12, c_22)


def centralMomentsFromDistributions (u, fin):
    ux = u[0, :, :]
    uy = u[1, :, :]
    (nx,ny) = ux.shape
    f = zeros((3,3,nx,ny))

    f[1,1] = fin[0]
    f[2,1] = fin[1]
    f[1,2] = fin[2]
    f[0,1] = fin[3]
    f[1,0] = fin[4]
    f[2,2] = fin[5]
    f[0,2] = fin[6]
    f[0,0] = fin[7]
    f[2,0] = fin[8]
    c_0_0 = f[0,0]              + f[0,1]           + f[0,2]
    c_0_1 = f[0,0]*(-1 - uy)    - f[0,1]*uy        + f[0,2]*(1 - uy)
    c_0_2 = f[0,0]*(-1 - uy)**2 + f[0,1]*uy**2     + f[0,2]*(1 - uy)**2
    c_1_0 = f[1,0]              + f[1,1]           + f[1,2]
    c_1_1 = f[1,0]*(-1 - uy)    - f[1,1]*uy        + f[1,2]*(1 - uy)
    c_1_2 = f[1,0]*(-1 - uy)**2 + f[1,1]*uy**2     + f[1,2]*(1 - uy)**2
    c_2_0 = f[2,0]              + f[2,1]           + f[2,2]
    c_2_1 = f[2,0]*(-1 - uy)    - f[2,1]*uy        + f[2,2]*(1 - uy)
    c_2_2 = f[2,0]*(-1 - uy)**2 + f[2,1]*uy**2     + f[2,2]*(1 - uy)**2
    c_00 = c_0_0              + c_1_0           + c_2_0
    c_10 = c_0_0*(-1 - ux)    - c_1_0*ux        + c_2_0*(1 - ux)
    c_20 = c_0_0*(-1 - ux)**2 + c_1_0*ux**2     + c_2_0*(1 - ux)**2
    c_01 = c_0_1              + c_1_1           + c_2_1
    c_11 = c_0_1*(-1 - ux)    - c_1_1*ux        + c_2_1*(1 - ux)
    c_21 = c_0_1*(-1 - ux)**2 + c_1_1*ux**2     + c_2_1*(1 - ux)**2
    c_02 = c_0_2              + c_1_2           + c_2_2
    c_12 = c_0_2*(-1 - ux)    - c_1_2*ux        + c_2_2*(1 - ux)
    c_22 = c_0_2*(-1 - ux)**2 + c_1_2*ux**2     + c_2_2*(1 - ux)**2

    return (c_00, c_10, c_01, c_11, c_20, c_02, c_21, c_12, c_22)


def distributionsFromCentralMoments (u, c_00, c_10, c_01, c_11, c_20, c_02, c_21, c_12, c_22):
    ux = u[0, :, :]
    uy = u[1, :, :]
    (nx,ny) = ux.shape
    f = zeros((3,3,nx,ny))
    c_0_0 = (c_00*(ux**2 - ux) + c_10*(2*ux - 1) + c_20) * 0.5
    c_1_0 =  c_00*(1 - ux**2)  - c_10*2*ux       - c_20
    c_2_0 = (c_00*(ux**2 + ux) + c_10*(2*ux + 1) + c_20) * 0.5
    c_0_1 = (c_01*(ux**2 - ux) + c_11*(2*ux - 1) + c_21) * 0.5
    c_1_1 =  c_01*(1 - ux**2)  - c_11*2*ux       - c_21
    c_2_1 = (c_01*(ux**2 + ux) + c_11*(2*ux + 1) + c_21) * 0.5
    c_0_2 = (c_02*(ux**2 - ux) + c_12*(2*ux - 1) + c_22) * 0.5
    c_1_2 =  c_02*(1 - ux**2)  - c_12*2*ux       - c_22
    c_2_2 = (c_02*(ux**2 + ux) + c_12*(2*ux + 1) + c_22) * 0.5
    f[0, 0] = (c_0_0*(uy**2 - uy) + c_0_1*(2*uy - 1) + c_0_2) * 0.5
    f[0, 1] =  c_0_0*(1 - uy**2)  - c_0_1*2*uy       - c_0_2
    f[0, 2] = (c_0_0*(uy**2 + uy) + c_0_1*(2*uy + 1) + c_0_2) * 0.5
    f[1, 0] = (c_1_0*(uy**2 - uy) + c_1_1*(2*uy - 1) + c_1_2) * 0.5
    f[1, 1] =  c_1_0*(1 - uy**2)  - c_1_1*2*uy       - c_1_2
    f[1, 2] = (c_1_0*(uy**2 + uy) + c_1_1*(2*uy + 1) + c_1_2) * 0.5
    f[2, 0] = (c_2_0*(uy**2 - uy) + c_2_1*(2*uy - 1) + c_2_2) * 0.5
    f[2, 1] =  c_2_0*(1 - uy**2)  - c_2_1*2*uy       - c_2_2
    f[2, 2] = (c_2_0*(uy**2 + uy) + c_2_1*(2*uy + 1) + c_2_2) * 0.5

    return array((f[1,1], f[2,1], f[1,2], f[0,1], f[1,0], f[2,2], f[0,2], f[0,0], f[2,0]))


def centralMomentsFromDistributions_min (u, fin):
    ux = u[0, :, :]
    uy = u[1, :, :]
    (nx,ny) = ux.shape
    f = zeros((3,3,nx,ny))
    uq = uy**2
    emu = (1 - uy)
    emuq = emu**2
    memu = (-1 - uy)
    memuq = memu**2

    f[1,1] = fin[0]
    f[2,1] = fin[1]
    f[1,2] = fin[2]
    f[0,1] = fin[3]
    f[1,0] = fin[4]
    f[2,2] = fin[5]
    f[0,2] = fin[6]
    f[0,0] = fin[7]
    f[2,0] = fin[8]
    c_0_0 = f[0,0]       + f[0,1]     + f[0,2]
    c_0_1 = f[0,0]*memu  - f[0,1]*uy  + f[0,2]*emu
    c_0_2 = f[0,0]*memuq + f[0,1]*uq  + f[0,2]*emuq
    c_1_0 = f[1,0]       + f[1,1]     + f[1,2]
    c_1_1 = f[1,0]*memu  - f[1,1]*uy  + f[1,2]*emu
    c_1_2 = f[1,0]*memuq + f[1,1]*uq  + f[1,2]*emuq
    c_2_0 = f[2,0]       + f[2,1]     + f[2,2]
    c_2_1 = f[2,0]*memu  - f[2,1]*uy  + f[2,2]*emu
    c_2_2 = f[2,0]*memuq + f[2,1]*uq  + f[2,2]*emuq

    memux = (-1 - ux)
    emux = (1 - ux)

    c_20 = c_0_0*memux**2 + c_1_0*ux**2     + c_2_0*emux**2
    c_11 = c_0_1*memux    - c_1_1*ux        + c_2_1*emux
    c_02 = c_0_2              + c_1_2           + c_2_2

    return (c_11, c_20, c_02)


def distributionsFromCentralMoments_min (u, c_00, c_11, c_20, c_02, c_22):
    ux = u[0, :, :]
    uy = u[1, :, :]
    (nx,ny) = ux.shape
    f = zeros((3,3,nx,ny))

    uxq = ux**2
    zux = 2*ux
    uxqmux = (uxq - ux)
    uxqpux = (uxq + ux)
    emuxq = (1 - uxq)

    c_0_0 = (c_00*uxqmux  + c_20) * 0.5
    c_1_0 =  c_00*emuxq   - c_20
    c_2_0 = (c_00*uxqpux  + c_20) * 0.5
    c_0_1 = ( c_11*(zux - 1) ) * 0.5
    c_1_1 =  - c_11*zux
    c_2_1 = ( c_11*(zux + 1) ) * 0.5
    c_0_2 = (c_02*uxqmux + c_22) * 0.5
    c_1_2 =  c_02*emuxq  - c_22
    c_2_2 = (c_02*uxqpux + c_22) * 0.5

    uyq = uy**2
    zuy = 2*uy

    uyqmuy = (uyq - uy)
    zuyme = (zuy - 1)
    emuyq = (1 - uyq)
    zuype = (zuy + 1)
    uyqpuy = (uyq + uy)

    f[0, 0] = (c_0_0*uyqmuy + c_0_1*zuyme + c_0_2) * 0.5
    f[0, 1] =  c_0_0*emuyq  - c_0_1*zuy   - c_0_2
    f[0, 2] = (c_0_0*uyqpuy + c_0_1*zuype + c_0_2) * 0.5
    f[1, 0] = (c_1_0*uyqmuy + c_1_1*zuyme + c_1_2) * 0.5
    f[1, 1] =  c_1_0*emuyq  - c_1_1*zuy   - c_1_2
    f[1, 2] = (c_1_0*uyqpuy + c_1_1*zuype + c_1_2) * 0.5
    f[2, 0] = (c_2_0*uyqmuy + c_2_1*zuyme + c_2_2) * 0.5
    f[2, 1] =  c_2_0*emuyq  - c_2_1*zuy   - c_2_2
    f[2, 2] = (c_2_0*uyqpuy + c_2_1*zuype + c_2_2) * 0.5

    return array((f[1,1], f[2,1], f[1,2], f[0,1], f[1,0], f[2,2], f[0,2], f[0,0], f[2,0]))


def normalizedCumulantsFromCentralMoments (rho, u, c_00, c_10, c_01, c_11, c_20, c_02, c_21, c_12, c_22):

    ux = u[0, :, :]
    uy = u[1, :, :]

    K_22 = c_22 - 2*c_11*c_11/rho - c_20*c_02/rho

    return ( log(rho) * rho, ux * rho, uy * rho, c_11, c_20, c_02, c_21, c_12, K_22)


def centralMomentsFromNormalizedCumulants (rho, K_00, K_10, K_01, K_11, K_20, K_02, K_21, K_12, K_22):
    # Transformation to central moments
    c_22 = K_22 + 2*K_11*K_11/rho + K_20*K_02/rho

    return (rho, 0, 0, K_11, K_20, K_02, K_21, K_12, c_22)


def centralMomentsFromNormalizedCumulants_min (rho, K_11, K_20, K_02):
    # Transformation to central moments
    c_22 = 2*K_11*K_11/rho + K_20*K_02/rho

    return (rho, K_11, K_20, K_02, c_22)



def normalizedCumulantsFromDistributions(rho, u, fin):
    centralMoments = centralMomentsFromDistributions (u, fin)

    return normalizedCumulantsFromCentralMoments(rho, u, *centralMoments)

def distributionsFromNormalizedCumulants(rho, u, *normalizedCumulants):
    centralMoments = centralMomentsFromNormalizedCumulants (rho, *normalizedCumulants)

    return distributionsFromCentralMoments (u, *centralMoments)

def normalizedCumulantsFromDistributions_min(rho, u, fin):
    return centralMomentsFromDistributions_min (u, fin)

def distributionsFromNormalizedCumulants_min(rho, u, *normalizedCumulants):
    centralMoments = centralMomentsFromNormalizedCumulants_min (rho, *normalizedCumulants)

    return distributionsFromCentralMoments_min (u, *centralMoments)
