import numpy
import scipy.linalg

A = numpy.array([[5, 3, 2, 0, 0, 0],[5, 8, 0, 3, 0, 0],[1, 0, 8, 4, 4, 0],[0, 2, 4, 4, 0, 5],[0, 0, 3, 0, 6, 9],[0, 0, 0, 4, 2, 7]])
numpy.set_printoptions(linewidth=200)
x = numpy.array([-7.0, 13, 9, -4, -.7, 42])
print(numpy.dot(A,x))
LU, piv = scipy.linalg.lu_factor(A)
print(LU)
print(piv)
