import numpy as np
import scipy.stats

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

gamma = 0.95
fac1 = scipy.stats.norm.ppf((1-gamma)/2, 0, 1)  
fac2 = scipy.stats.norm.ppf((1-gamma)/2 + gamma, 0, 1)

print(fac1)
print(fac2)