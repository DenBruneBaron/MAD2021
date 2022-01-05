import numpy as np
import scipy
from scipy.stats import t

# v = np.array([-1,1])
# print(v)

# matrix = np.array([[2, 0.8], [0.8, 0.6]])
# print(matrix)

# multi_1 = np.dot(matrix, v)
# print("1st multiplication: ",multi_1)

# multi_2 = np.dot(matrix, multi_1)
# print("2nd multiplication: ",multi_2)

# multi_3 = np.dot(matrix, multi_2)
# print("3rd multiplication: ",multi_3)

# multi_4 = np.dot(matrix, multi_3)
# print("4th multiplication: ", multi_4)

#

# Mean of my sample
mean = 0.6

# Samples (5 flowers)
n_samples = 5

# I need to divide alpha since I'm making use of the two-side test
alpha_val = 0.05 / 2
std_deviation = 0.7416

# Performing the t.ppf, in order to find c1 and c2 
c1 = t.ppf(alpha_val, n_samples-1, loc= mean, scale = std_deviation)
c2 = t.ppf((1 - alpha_val), n_samples-1, loc= mean, scale = std_deviation)

print("c1: lower_cutoff", c1)
print("c2: upper_cutoff", c2)

M = np.array([[1,2,3],[4,5,6]])
print(M.shape)