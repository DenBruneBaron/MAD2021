import numpy as np

nd1_array = np.array([1,2,3,4,5,6,7,8,9]) # 1 dimension
print(nd1_array.shape)
print(len(nd1_array))

nd2_array = np.array([[1,2,3,0],[4,5,6,0],[7,8,9,0],[10,11,12,0]]) # two dimension
print(nd2_array.shape)
print(len(nd2_array))

reshape_nd2 = nd2_array.reshape((len(nd2_array), -1))
print("reshape")
print(reshape_nd2.shape)
print(reshape_nd2)