import linregLOOCV
import numpy as np
import matplotlib.pyplot as plt

# Reads the data from the file
raw = np.genfromtxt('men-olympics-100.txt', delimiter=' ')
transposed = raw.T

# Extract the first "row", (index 1, since the array is 0-indexed) from raw
OL_year = raw[:,0].T

# Extract the second "row", (index 1, since the array is 0-indexed) from raw
OL_run_times = raw[:,1].T

# Create lamda values for LOOCV
lambda_values = np.logspace(-8, 0, 100, base=10)
print("labdas shape", lambda_values.shape)
print("lambdas:", lambda_values)
#print(lambda_values)
print()
lambda_range = np.insert(lambda_values, 0, .0) # prepend lamba = 0
print("lambda_range shape", lambda_range.shape)
print("lambdas:", lambda_range)