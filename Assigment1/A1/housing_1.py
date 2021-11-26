import numpy as np
import math
import matplotlib as plt
from matplotlib import pyplot as ppl

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
#print(t_train)

# (a) compute mean of prices on training set

#print(t_train.shape)
house_prices = t_train
#print(len(house_prices))
#for i in house_prices:
#  print(i)
simple_mean_result = house_prices.mean()
print("naive average price:", simple_mean_result)

# (b) RMSE function
def RMSE(t,tp):
    res_MSE = np.square(np.subtract(t,tp)).mean() 
    RSM = math.sqrt(res_MSE)
    print("result of RSME:", RSM)

RMSE(t_test, simple_mean_result)

# (c) visualization of results

x = t_test
y = np.full((253,1), simple_mean_result)
ppl.scatter(x,y)
ppl.show()

class RMSE_Function():
    """
    Linear regression implementation.
    """

    def __init__(self):
        
        pass
            
    def RMSE(t,tp):
        res_MSE = np.square(np.subtract(t,tp)).mean() 
        RSM = math.sqrt(res_MSE)
        print("result of RSME:", RSM)