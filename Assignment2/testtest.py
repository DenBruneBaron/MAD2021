import linweighreg
import numpy as np
import matplotlib.pyplot as plt

# Reads the data from the file
raw = np.genfromtxt('men-olympics-100.txt', delimiter=' ')
transposed = raw.T

# Extract the first "row", (index 1, since the array is 0-indexed) from raw
OL_year = raw[:,0]

# Extract the second "row", (index 1, since the array is 0-indexed) from raw
OL_run_times = raw[:,1]

# Create lamda values for LOOCV
lambda_values = np.logspace(-8, 0, 100, base=10)


#Olympic years
x = OL_year 

#First place values
y = OL_run_times

idM = np.identity(len(x))

n = x.shape[0]
x = np.array(x).reshape((n, -1))
# prepend a column of ones
ones = np.ones((x.shape[0], 1))
x = np.concatenate((ones, x), axis=1)


y = np.array(y).reshape((n, -1))
# prepend a column of ones
ones = np.ones((y.shape[0], 1))
y = np.concatenate((ones, y), axis=1)

#print("x ", x)
#print("y ", y)
#print("lambdas", lambda_values)
print("x shape", x.shape)
print("y shape", y.shape)
print("lambda shape", lambda_values.shape)
print("identity", idM.shape)
print()
print(type(x))
print(type(y))
print(type(lambda_values))

