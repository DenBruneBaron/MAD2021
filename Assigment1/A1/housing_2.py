import numpy as np
import pandas as pd
import linreg
import math
import matplotlib.pyplot as plt

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


# (b) fit linear regression using only the first feature
print("Ex (b)")
model_single = linreg.LinearRegression()
model_single.fit(X_train[:,0], t_train)
model_single.predict(X_train[:,0])



# (c) fit linear regression model using all features
print("Ex (c)")
model_single = linreg.LinearRegression()
model_single.fit(X_train, t_train)
model_single.predict(X_train)


# (d) evaluation of results
print("Ex (d) - ONLY FIRST VALUE")
model_single = linreg.LinearRegression()
model_single.fit(X_test[:,0], t_train)
model_single.predict(X_test[:,0])
d = model_single.predict(X_test[:,0])
print("Ex (d) - ALL VALUES")
model_single.fit(X_test, t_train)
f = model_single.predict(X_test)

def RMSE(t,tp):
    res_MSE = np.square(np.subtract(t,tp)).mean() 
    RSM = math.sqrt(res_MSE)
    print("result of RSME:", RSM)

RMSE(t_test, d)
RMSE(t_test, f)