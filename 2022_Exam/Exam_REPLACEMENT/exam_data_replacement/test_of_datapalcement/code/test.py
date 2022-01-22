import numpy as np
import filecmp
# # seed = 111
# # x = np.random.default_rng(seed)
# # print(x)



# a = np.arange(9).reshape(3,3)
# flatten = a.flat[::a.shape[1]+1]
# print(flatten)


# M = np.zeros(9).reshape(3,3)
# diag_values = np.array([0.00001, 0.00001, 1.0])
# np.fill_diagonal(M, diag_values)

# print(M)

  
f1 = "./accent-mfcc-data_shuffled_train.txt"
f2 = "D:\\Uddannelse\\Datalogi\\KU\\2_aar\\MAD\\MAD2021\\Joachim\\exam\\code\\data\\accent-mfcc-data_shuffled_train.txt"

f3 = "./accent-mfcc-data_shuffled_train.txt"
f4 = "D:\\Uddannelse\\Datalogi\\KU\\2_aar\\MAD\\MAD2021\\Joachim\\exam\\code\\data\\accent-mfcc-data_shuffled_train.txt"

# shallow comparison
result = filecmp.cmp(f1, f2)
print(result)
# deep comparison
result = filecmp.cmp(f1, f2, shallow=False)
print(result)

result = filecmp.cmp(f3, f4)
print(result)
# deep comparison
result = filecmp.cmp(f3, f4, shallow=False)
print(result)
