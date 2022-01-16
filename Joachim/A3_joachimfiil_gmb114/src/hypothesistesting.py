#!/Usr/bin/env python
#
# # Exercise 3: Hypothesis Testing 
#
# 5.12.2020
#
# Joachim Fiil
# gmb114@alumni.ku.dk

import numpy as np
import scipy.stats

# flowering times (in days)
n = 5
x = np.array([4.1,4.8,4.0,4.5,4.0]) # replicate 1 without knockout
y = np.array([3.1,4.3,4.5,3.0,3.5]) # replicate 2 with knockout
z = x-y # difference in flowering time


# pretty printing the sample data
row_header_format = "| %-30s "
cell_format = "| %3.1f "

print(row_header_format % "Plant", end='')
for i in range(n):
    print("| %3d " %(i+1), end='')
print("|")
print(row_header_format % "Replicate 1 without knockout", end='')
for i in range(n):
    print(cell_format %x[i], end='')
print("|")
print(row_header_format % "Replicate 2 with knockout", end='')
for i in range(n):
    print(cell_format % y[i], end='')
print("|")
print(row_header_format % "Difference in flowering time", end='')
for i in range(n):
    print(cell_format % z[i], end='')
print("|")

print("Assuming, that (the difference) z is Normal distributed with mean = 0")

# sample mean
zmean = np.mean(z)
print("The sample mean of z: %.4f" %zmean)

# sample standard deviation
s = np.sqrt(np.var(z, ddof=1))
print("The sample standard deviation s: %.4f" %s)

# t-test
mu = 0
alpha = 0.05
t = np.sqrt(n)*(zmean - mu) / s
print("Test statistics for the samples t: %.4f" %t)

# critical value
c1 = scipy.stats.t.ppf(alpha/2, n-1)
c2 = scipy.stats.t.ppf(1-alpha/2, n-1)
print("Critical value c1: %.4f" %c1)
print("Critical value c2: %.4f" %c2)
