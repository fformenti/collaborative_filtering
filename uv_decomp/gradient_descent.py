import numpy as np
import pandas as pd


Y = np.array([[5, 1, 4, 5, 1], [0, 5, 2, 1, 4], [1, 4, 1, 1, 2],\
              [4, 1, 5, 5, 4], [5, 3, 4, 5, 4], [1, 5, 1, 1, 1], \
              [5, 1,0, 5, 4]])

k = 2


U = np.random.random((len(Y),k))
V = np.random.random((len(Y[0]),k))

R = Y.clip(max=1)

N = np.count_nonzero(Y)

eta = 0.05
error = 0.0
min_error = 0.0001

while True:
    U = U + (2.0/N) * eta * np.dot(np.multiply(Y - np.dot(U, V.transpose()), R), V)
    V = V + (2.0/N) * eta * np.dot(np.multiply(Y - np.dot(U, V.transpose()), R).transpose(), U)

    M = np.multiply((Y - np.dot(U, V.transpose())), R)
    new_error = np.sum(np.square(M)) / N

    if abs(error - new_error) < min_error:
        break
    error = new_error

print "error" 
print error
print "U:" 
print U
print "V:"
print V


