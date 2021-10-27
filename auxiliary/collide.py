from numpy import *

from auxiliary.LBMHelpers import noslip
from auxiliary.transforms import *

import sys


def BGKCollide(fin, feq, omega):
    return fin - omega * (fin - feq)


def TRTCollide(fin, feq, omega_plus, magicNumber): #hint 3/16 = 0.1875
    f_plus = 0.5 * (fin[:,:,:] + fin[noslip[:],:,:])
    f_minus = 0.5 * (fin[:,:,:] - fin[noslip[:],:,:])
    feq_plus = 0.5 * (feq[:,:,:] + feq[noslip[:],:,:])
    feq_minus = 0.5 * (feq[:,:,:] - feq[noslip[:],:,:])
    omega_minus = (4.0 - 2.0 * omega_plus ) / (4.0 * magicNumber * omega_plus + 2.0 - omega_plus)

    return fin - omega_plus * (f_plus - feq_plus) - omega_minus * (f_minus - feq_minus)


def cumulantCollide(fin, rho, u, omega):
    (K_00, K_10, K_01, K_11, K_20, K_02, K_21, K_12, K_22) = normalizedCumulantsFromDistributions (rho, u, fin)

    #print "\ncollide"
    # collision
    K_00_p = K_00
    K_10_p = K_10
    K_01_p = K_01

    K_11_p = (1-omega)*K_11

    K_20_p = rho/3. + 0.5*(1-omega)*(K_20 - K_02)
    K_02_p = rho/3. - 0.5*(1-omega)*(K_20 - K_02)

    K_21_p = 0
    K_12_p = 0

    K_22_p = 0

    return distributionsFromNormalizedCumulants (rho, u, K_00_p, K_10_p, K_01_p, K_11_p, K_20_p, K_02_p, K_21_p, K_12_p, K_22_p)


def cumulantCollide_min(fin, rho, u, omega):
    (K_11, K_20, K_02) = normalizedCumulantsFromDistributions_min (rho, u, fin)

    K_11_p = (1-omega)*K_11

    K_20_p = rho/3. + 0.5*(1-omega)*(K_20 - K_02)
    K_02_p = rho/3. - 0.5*(1-omega)*(K_20 - K_02)

    return distributionsFromNormalizedCumulants_min (rho, u, K_11_p, K_20_p, K_02_p)


def cumulantCollideAll(fin, rho, u, omega1, omega2, omega3, omega4):
    (K_00, K_10, K_01, K_11, K_20, K_02, K_21, K_12, K_22) = normalizedCumulantsFromDistributions (rho, u, fin)

    # collision
    K_00_p = K_00
    K_10_p = K_10
    K_01_p = K_01

    K_11_p = (1-omega1)*K_11

    K_20_p = 0.5 * (K_20 + K_02 + omega2*(2/3 - K_20 - K_02) + (1-omega1)*(K_20 - K_02))
    K_02_p = 0.5 * (K_20 + K_02 + omega2*(2/3 - K_20 - K_02) - (1-omega1)*(K_20 - K_02))

    K_21_p = (1-omega3)*K_21
    K_12_p = (1-omega3)*K_12

    K_22_p = (1-omega4)*K_22

    return distributionsFromNormalizedCumulants (rho, u, K_00_p, K_10_p, K_01_p, K_11_p, K_20_p, K_02_p, K_21_p, K_12_p, K_22_p)

def cumulantCollideAll_new(fin, rho, u, omega1, omega2, omega3, omega4):
    (K_00, K_10, K_01, K_11, K_20, K_02, K_21, K_12, K_22) = normalizedCumulantsFromDistributions (rho, u, fin)

    # collision
    K_00_p = K_00
    K_10_p = K_10
    K_01_p = K_01

    K_11_p = (1-omega1)*K_11

    K_20_p = 0.5 * (K_20 + K_02 + omega2*(2/3*rho - K_20 - K_02) + (1-omega1)*(K_20 - K_02))
    K_02_p = 0.5 * (K_20 + K_02 + omega2*(2/3*rho - K_20 - K_02) - (1-omega1)*(K_20 - K_02))

    K_21_p = (1-omega3)*K_21
    K_12_p = (1-omega3)*K_12

    K_22_p = (1-omega4)*K_22

    return distributionsFromNormalizedCumulants (rho, u, K_00_p, K_10_p, K_01_p, K_11_p, K_20_p, K_02_p, K_21_p, K_12_p, K_22_p)


def centralMomentSRT(fin, feq, u, omega):
    # central moments
    (c_00, c_10, c_01, c_11, c_20, c_02, c_21, c_12, c_22) = centralMomentsFromDistributions (u, fin)

    # transform the equilibrium function
    (c_00_eq, c_10_eq, c_01_eq, c_11_eq, c_20_eq, c_02_eq, c_21_eq, c_12_eq, c_22_eq) = centralMomentsFromDistributions (u, feq)

    # collision
    c_00_p = c_00 + omega*(c_00_eq - c_00)
    c_10_p = c_10 + omega*(c_10_eq - c_10)
    c_01_p = c_01 + omega*(c_01_eq - c_01)

    c_11_p = c_11 + omega*(c_11_eq - c_11)

    c_20_p = c_20 + omega*(c_20_eq - c_20)
    c_02_p = c_02 + omega*(c_02_eq - c_02)

    c_21_p = c_21 + omega*(c_21_eq - c_21)
    c_12_p = c_12 + omega*(c_12_eq - c_12)

    c_22_p = c_22 + omega*(c_22_eq - c_22)

    return distributionsFromCentralMoments (u, c_00_p, c_10_p, c_01_p, c_11_p, c_20_p, c_02_p, c_21_p, c_12_p, c_22_p)

def centralMoment(fin, feq,rho, u, omega1,omega2,omega3,omega4):
    # central moments
    (c_00, c_10, c_01, c_11, c_20, c_02, c_21, c_12, c_22) = centralMomentsFromDistributions (u, fin)

    # transform the equilibrium function
    (c_00_eq, c_10_eq, c_01_eq, c_11_eq, c_20_eq, c_02_eq, c_21_eq, c_12_eq, c_22_eq) = centralMomentsFromDistributions (u, feq)

    # collision
    c_00_p = c_00
    c_10_p = c_10
    c_01_p = c_01

    c_11_p = c_11 + omega1*(c_11_eq - c_11)

    c_20_p = 0.5 * (c_20 + c_02 + omega2*(2*rho/3 - c_20 - c_02) + (1-omega1)*(c_20 - c_02))
    c_02_p = 0.5 * (c_20 + c_02 + omega2*(2*rho/3 - c_20 - c_02) - (1-omega1)*(c_20 - c_02))

    c_21_p = c_21 + omega3*(c_21_eq - c_21)
    c_12_p = c_12 + omega3*(c_12_eq - c_12)

    c_22_p = c_22 + omega4*(c_22_eq - c_22)

    return distributionsFromCentralMoments (u, c_00_p, c_10_p, c_01_p, c_11_p, c_20_p, c_02_p, c_21_p, c_12_p, c_22_p)


def cumulantCollideAllInOne(fin, rho, u, omega):
    ux = u[0, :]
    uy = u[1, :]

    uyq = uy**2
    uxq = ux**2
    zux = 2*ux
    zuy = 2*uy

    emuy = (1 - uy)
    emuyq = emuy**2

    memuy = (-1 - uy)
    memuyq = memuy**2

    uxqqmux = (uxq - ux)
    uxqqpux = (uxq + ux)
    uyqqmuy = (uyq - uy)
    uyqqpuy = (uyq + uy)

    emuxqq = (1 - uxq)
    emuyqq = (1 - uyq)

    zuyme = (zuy - 1)
    zuype = (zuy + 1)

    c_0_0 = fin[7]        + fin[3]     + fin[6]
    c_0_1 = fin[7]*memuy  - fin[3]*uy  + fin[6]*emuy
    c_0_2 = fin[7]*memuyq + fin[3]*uyq + fin[6]*emuyq
    c_1_0 = fin[4]        + fin[0]     + fin[2]
    c_1_1 = fin[4]*memuy  - fin[0]*uy  + fin[2]*emuy
    c_1_2 = fin[4]*memuyq + fin[0]*uyq + fin[2]*emuyq
    c_2_0 = fin[8]        + fin[1]     + fin[5]
    c_2_1 = fin[8]*memuy  - fin[1]*uy  + fin[5]*emuy
    c_2_2 = fin[8]*memuyq + fin[1]*uyq + fin[5]*emuyq

    memux = (-1 - ux)
    emux = (1 - ux)

    c_20 = c_0_0*memux**2 + c_1_0*ux**2     + c_2_0*emux**2
    c_11 = c_0_1*memux    - c_1_1*ux        + c_2_1*emux
    c_02 = c_0_2          + c_1_2           + c_2_2

    #now, we have cumulants
    emo = (1-omega)
    rhobyt = rho/3.
    K_11_p = emo*c_11
    relaxedDiff = 0.5*emo*(c_20 - c_02)

    K_20_p = rhobyt + relaxedDiff
    K_02_p = rhobyt - relaxedDiff

    #back transformation
    c_22 = 2*K_11_p*K_11_p/rho + K_20_p*K_02_p/rho

    c_0_0 = (rho*uxqqmux  + K_20_p) * 0.5
    c_1_0 =  rho*emuxqq   - K_20_p
    c_2_0 = (rho*uxqqpux  + K_20_p) * 0.5
    c_0_1 = ( K_11_p*(zux - 1) ) * 0.5
    c_1_1 =  - K_11_p*zux
    c_2_1 = ( K_11_p*(zux + 1) ) * 0.5
    c_0_2 = (K_02_p*uxqqmux + c_22) * 0.5
    c_1_2 =  K_02_p*emuxqq  - c_22
    c_2_2 = (K_02_p*uxqqpux + c_22) * 0.5

    fin[7] = (c_0_0*uyqqmuy + c_0_1*zuyme + c_0_2) * 0.5
    fin[3] =  c_0_0*emuyqq  - c_0_1*zuy   - c_0_2
    fin[6] = (c_0_0*uyqqpuy + c_0_1*zuype + c_0_2) * 0.5
    fin[4] = (c_1_0*uyqqmuy + c_1_1*zuyme + c_1_2) * 0.5
    fin[0] =  c_1_0*emuyqq  - c_1_1*zuy   - c_1_2
    fin[2] = (c_1_0*uyqqpuy + c_1_1*zuype + c_1_2) * 0.5
    fin[8] = (c_2_0*uyqqmuy + c_2_1*zuyme + c_2_2) * 0.5
    fin[1] =  c_2_0*emuyqq  - c_2_1*zuy   - c_2_2
    fin[5] = (c_2_0*uyqqpuy + c_2_1*zuype + c_2_2) * 0.5

    return fin
