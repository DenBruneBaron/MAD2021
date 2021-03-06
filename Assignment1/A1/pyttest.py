import numpy as np
from numpy.core.fromnumeric import reshape

X = np.array([[0.5,37.6],[2.3,39.1],[2.9,36.2]])
t = np.array([3.2, 1.9, 1])
print("Orignial matrix:")
print(np.matrix(X))
print()
print("Transposed matrix:")
print(np.matrix(X.T))
print()
adding_ones = np.insert(X,0,1,axis=1)
print("adding value of 1 to the matrix: extending with one column to the left, with ones")
print(adding_ones)
print()
print("Transposed matrix: printing X^T")
print(np.matrix(adding_ones.T))
print()
print("Dotted matrix matrix: printing X^TX")
var1 = np.dot(adding_ones.T,adding_ones)
print(np.matrix(var1))
print()
inverse_of_matrix = np.linalg.inv(var1)
print("inverted matrix: printing (X^TX)^-1")
print(inverse_of_matrix)
print()
print("inverted matrix: printing (X^TX)^-1X^T")
rdy_to_dot_with_vector = np.dot(inverse_of_matrix, adding_ones.T)
print(rdy_to_dot_with_vector)
#reshaped = np.array(rdy_to_dot_with_vector).reshape((len(rdy_to_dot_with_vector), -1))
#print(reshaped)
#print(t.shape)
result = np.dot(rdy_to_dot_with_vector,t)
print()
print(result)
