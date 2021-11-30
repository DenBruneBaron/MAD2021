import numpy as np
from matplotlib import pyplot as plt

# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]

# make sure that we have N-dimensional Np arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
t_prediction = np.mean(t_train)
print("a) The mean of prices on the training set: %f" %t_prediction)

# (b) RMSE function
def rmse(t, tp):
    return np.sqrt(np.mean((t-tp)**2))

print("b) The root-mean-square error: %f" %rmse(t_test, t_prediction))

# (c) visualization of results
t_prediction = np.full((len(t_train), 1), np.mean(t_train))
plt.scatter(t_test, t_prediction)
plt.xlabel('True House Prices')
plt.ylabel('Estimates')
#plt.savefig('figures/ex3_c.png')