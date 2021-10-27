#!/usr/bin/python
from scipy import misc
import numpy


def importImage(path):
    image = misc.imread(path)
    image = image[:, :, 0]
    image = image < 50.
    image = numpy.swapaxes(image, 1, 0)
    return image


def flipY(image):
    return numpy.fliplr(image)


def setSubmatrixAt(matrix, submatrix, startX, startY):
    dx = submatrix.shape[0]
    dy = submatrix.shape[1]
    matrix[startX: startX + dx, startY: startY+dy] = submatrix
    return matrix

if __name__ == "__main__":
    import sys
    importImage(sys.argv[1])
