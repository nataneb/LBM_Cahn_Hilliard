from auxiliary.LBMHelpers import iLeft, iCentV, iRight, iTop, iCentH, iBot
from auxiliary.LBMHelpers_CHE import t

def YuLeft(f, feq):
    f[iRight, :] = feq[iLeft, :] + (feq[iRight, :] - f[iLeft, :])
    
#Realization of neutral wetting boundary condition based on the equilibrium distribution function (11) (Zheng et al. 2015)
def NeutralWetting(f, upper = False, lower = False, left = False, right = False):
    (q, nx, ny) = f.shape
    if upper:
        H_temp = (f[2,:,ny-1]+f[5,:,ny-1]+f[6,:,ny-1])/(t[4]+t[7]+t[8])
        f[4,:, ny-1] = t[4]*H_temp
        f[7,:, ny-1] = t[7]*H_temp
        f[8,:, ny-1] = t[8]*H_temp
    if lower:
        H_temp = (f[4,:,0]+f[7,:,0]+f[8,:,0])/(t[2]+t[5]+t[6])
        f[2,:, 0] = t[2]*H_temp
        f[5,:, 0] = t[5]*H_temp
        f[6,:, 0] = t[6]*H_temp

    

