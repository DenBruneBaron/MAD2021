#!/usr/bin/env python
#
# MAD 2020-21, Exam
# Question 2 (Hypothesis Testing)
#

import numpy as np
import scipy.stats

# known variance
sigma_squared = 1.0

# samples
X = np.array([8.2, 7.9, 8.7, 8.3, 8.5, 8.3, 8.8, 8.2, 8.7, 7.6, 8.4])
n = len(X)

# sample mean
X_mean = np.mean(X)

# z-test
mu = 8.5
alpha = 0.05
z = np.sqrt(n) * (X_mean - mu) / np.sqrt(sigma_squared)
c = scipy.stats.norm.ppf(alpha)

print("Known variance: %.1f" %sigma_squared)
print("Samples:", X)
print("Number of samples:", n)
print("Sample mean x: %.4f" %X_mean)
print("Significance Level alpha: %.2f" %alpha)
print("Test statistics for the samples z: %.4f" %z)
print("Critical Value: %.4f" %c)
