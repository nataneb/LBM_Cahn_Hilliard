#!/usr/bin/python
from pyevtk.hl import gridToVTK
from numpy import *


def saveToVTK(velocity, rho, prefix, saveNumber, grid):
    name = "./" + prefix + "." + saveNumber
    gridToVTK(name, grid[0], grid[1], grid[2],
              pointData = {'velocity': velocity,
                           'pressure': rho})
