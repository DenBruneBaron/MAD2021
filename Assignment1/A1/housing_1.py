import numpy as np
from matplotlib import pyplot as plt

# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]

# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
house_prices = t_train
simple_mean_result = np.mean(house_prices)
print("a) mean of prices, using training set:", simple_mean_result)

# (b) RMSE function
def RMSE(t, tp):
    res = np.sqrt(np.square(np.subtract(t,tp)).mean())
    return(res)

print("b) RMSE: %f" %RMSE(t_test, simple_mean_result))

# (c) visualization of results
x = t_test
y = np.full((len(t_train),1), simple_mean_result)
plt.scatter(x,y)
plt.show()