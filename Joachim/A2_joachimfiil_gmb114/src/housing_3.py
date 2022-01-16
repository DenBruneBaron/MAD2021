'''
MAD 2020-21
Assignemt 2

Joachim Fiil, gmb114@alumni.ku.dk

Exercise 1: Weighted Average Loss

NOTE:
execute from inside src/ directory to handle path to data correctly

'''

import numpy as np
import linweighreg
import matplotlib.pyplot as plt

# load data
train_data = np.loadtxt("../data/boston_train.csv", delimiter=",")
test_data = np.loadtxt("../data/boston_test.csv", delimiter=",")
# extract features and  prices
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# ensure N-dimensional nparrays
t_train = t_train.reshape((X_train.shape[0], 1))
t_test = t_test.reshape((X_test.shape[0], 1))

print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

print("\n")
print("Linear weighted regression")
print("\n")

# fit linear weighted regression model using all features
# and diagonal matrix A with \alpha_n = t_n^2 in diagonal
A = np.diagflat(np.multiply(t_train, t_train))
model_single = linweighreg.LinearWeightedRegression()
model_single.fit(X_train[:,0], t_train, A)
model_all = linweighreg.LinearWeightedRegression()
model_all.fit(X_train, t_train, A)

print("\n")
print("Fitting on the first feature only")
for i in range(len(model_single.w)):
    print("\tw%i = %s" %(i, model_single.w[i]))


print("\n")
print("Fitting on all of the features")
for i in range(len(model_all.w)):
    print("\tw%i = %s" %(i, model_all.w[i]))

print("\n")
print("Evaluation of results")
print("\n")

pred_single = model_single.predict(X_test[:,0])
pred_all = model_all.predict(X_test)

# RMSE
print("\tRMSE for first feature fit: %f" %np.sqrt(np.mean((t_test - pred_single)**2)))
print("\tRMSE for all features fit: %f" %np.sqrt(np.mean((t_test - pred_all)**2)))

# plot: first feature only
plt.figure()
plt.scatter(t_test, pred_single)
plt.title('First feature fit (CRIM)')
plt.xlabel('True House Prices')
plt.ylabel('Estimates')
plt.savefig('../plots/ex1_linweighreg_single.png')

# plot: all features
plt.figure()
plt.scatter(t_test, pred_all)
plt.title('Fitting on all features')
plt.xlabel('True House Prices')
plt.ylabel('Estimates')
plt.savefig('../plots/ex1_linweighreg_all.png')
