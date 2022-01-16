import numpy as np
import pandas
import linreg
import matplotlib.pyplot as plt

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

# (b) fit linear regression using only the first feature (i.e. CRIM: per capita
# crime rate by town)
model_single = linreg.LinearRegression()
model_single.fit(X_train[:,0], t_train)

print("b) First feature fit (CRIM)")
print("\tw0 = %f" %model_single.w[0])
print("\tw1 = %f" %model_single.w[1])


# (c) fit linear regression model using all features

model_all = linreg.LinearRegression()
model_all.fit(X_train, t_train)

print("c) Fitting on all of the features")
for i in range(len(model_all.w)):
    print("\tw%i = %s" %(i, model_all.w[i]))

# (d) evaluation of results

print("d) Evaluation of results")
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
plt.savefig('figures/ex4_single.png')

# plot: all features
plt.figure()
plt.scatter(t_test, pred_all)
plt.title('Fitting on all features')
plt.xlabel('True House Prices')
plt.ylabel('Estimates')
plt.savefig('figures/ex4_all.png')
