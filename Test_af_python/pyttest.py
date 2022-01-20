import numpy as np


# X = np.array([[0.5,37.6],[2.3,39.1],[2.9,36.2]])
# print("Orignial matrix:")
# print(np.matrix(X))
# print("Transposed matrix:")
# print(np.matrix(X.T))
# adding_ones = np.insert(X,0,1,axis=1)
# print("adding value of 1 to the matrix: extending with one column to the left, with ones")
# print(adding_ones)
# print("Transposed matrix: printing X^T")
# print(np.matrix(adding_ones.T))
# print("Dotted matrix matrix: printing X^TX")
# var1 = np.dot(adding_ones.T,adding_ones)
# print(np.matrix(var1))
# inverse_of_matrix = np.linalg.inv(var1)
# print("inverted matrix: printing (X^TX)^-1")
# print(inverse_of_matrix)
# print("inverted matrix: printing (X^TX)^-1X^T")
# print(np.dot(inverse_of_matrix, adding_ones.T))


# vec= np.array([-33.3, -15.1])
# print(vec[0])
#def vecLen(x):
#    return np.sqrt((x[0])**2 + (x[1])**2)
print(np.sqrt((1.8204)**2 + (0.3546)**2))
# vecLen(vec)
#print(np.sqrt(1336.9))
#print((-33.3 / 36.56))
#print((-15.1 / 36.56))

print(1.8204 / 1.8546)
print(0.3546 / 1.8546)
print()
print(np.sqrt((0.3546 / 1.8546)**2 + (1.8204 / 1.8546)**2))