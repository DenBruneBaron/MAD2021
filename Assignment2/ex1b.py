import numpy as np
import matplotlib.pyplot as plt
import linweighreg

# loading data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]

# make sure that we have N-dimensional np arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])


# fit linear regression model using all features
print("Linear fit:")
model_all = linweighreg.LinearRegression()
model_all.fit(X_train, t_train)
all_values = model_all.w

# Prints all the weights from fit() 
for i in range(len(all_values)):
    print("\tw%i : %s" %(i, model_all.w[i]))

# evaluation of results
pred_all = model_all.predict(X_test)