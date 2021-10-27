from numpy import *

def stream(f):
    post = f.copy()
    post[0, :, :] = f[0, :, :]

    post[1, :, :] = roll(f[1, :, :],  1,  axis=0)
    post[2, :, :] = roll(f[2, :, :],  -1,  axis=1) # due to the array indices in python, this should be rolled -1
    post[3, :, :] = roll(f[3, :, :], -1,  axis=0)
    post[4, :, :] = roll(f[4, :, :], 1,  axis=1) # due to the array indices in python, this should be rolled +1

    post[5, :, :] = roll(roll(f[5, :, :],  1,  axis=0),  -1,  axis=1)
    post[6, :, :] = roll(roll(f[6, :, :], -1,  axis=0),  -1,  axis=1)
    post[7, :, :] = roll(roll(f[7, :, :], -1,  axis=0), 1,  axis=1)
    post[8, :, :] = roll(roll(f[8, :, :],  1,  axis=0), 1,  axis=1)

    return post

def stream2(f):
    post = f.copy()
    #print(f[5,10])
    post[0, :, :] = f[0, :, :]

    post[1, :, :] = roll(f[1, :, :],  1,  axis=0)
    post[2, :, :] = roll(f[2, :, :],  1,  axis=1) #just follow the velocity directions
    post[3, :, :] = roll(f[3, :, :], -1,  axis=0)
    post[4, :, :] = roll(f[4, :, :], -1,  axis=1) # due to the array indices in python, this should be rolled +1

    post[5, :, :] = roll(roll(f[5, :, :],  1,  axis=0),  1,  axis=1)
    post[6, :, :] = roll(roll(f[6, :, :], -1,  axis=0),  1,  axis=1)
    post[7, :, :] = roll(roll(f[7, :, :], -1,  axis=0), -1,  axis=1)
    post[8, :, :] = roll(roll(f[8, :, :],  1,  axis=0), -1,  axis=1)
    #print(post[5,10])
    return post
