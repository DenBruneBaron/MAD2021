#!/usr/bin/env python
#
# MAD 2020-21, EXAM
# Question 3 (Confidence Intervals)
#

import numpy as np
import scipy.stats

# samples from normal distributed X \sim N(mu, sigma_squared)
x = np.array([56.6, 59.0, 53.2, 66.1, 51.3, 50.4, 53.5, 44.5, 46.3, 60.3])
n = len(x) # number of samples

print("Number of samples n: ", n)

# sample mean
x_mean = np.mean(x)

#
# a)
#
# known variance
sigma_squared = 5.0**2

gamma = 0.95
c = scipy.stats.norm.ppf((1+gamma)/2)

ac = x_mean - c * sigma_squared / np.sqrt(n)
bc = x_mean + c * sigma_squared / np.sqrt(n)

print("a)")
print("Known variance: %.1f" %sigma_squared)
print("Sample mean: %.2f" %x_mean)
print("Critical value c: %.2f" %c)
print("Confidence Interval: [ %.2f ; %.2f ]" %(ac, bc))

#
# b)
#
# sample variance
S = np.mean(np.var(x, ddof=1))

gamma = 0.95
c = scipy.stats.t.ppf((1+gamma)/2, n-1)

ac = x_mean - c * sigma_squared / np.sqrt(n)
bc = x_mean + c * sigma_squared / np.sqrt(n)

print("b)")
print("Sample variance S: %.2f" %S)
print("Critical value c: %.2f" %c)
print("Confidence Interval: [ %.2f ; %.2f ]" %(ac, bc))
