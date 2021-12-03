import numpy as np
import pandas
import linreg
import matplotlib.pyplot as plt

# load data
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

def RMSE(t,tp):
    res = np.sqrt(np.square(np.subtract(t,tp)).mean())
    print(res)
    return(res)

# (b) fit linear regression using only the first feature
# Crime rate by town
print("Ex (b)")
model_single = linreg.LinearRegression()
model_single.fit(X_train[:,0], t_train)
print(model_single.w[0], model_single.w[1])
#print(model_single.w[1])


# (c) fit linear regression model using all features
print("Ex (c)")
model_all = linreg.LinearRegression()
model_all.fit(X_train, t_train)
all_values = model_all.w

for i in range(len(all_values)):
    print("\tw%i = %s" %(i, model_all.w[i]))

# (d) evaluation of results
pred_single = model_single.predict(X_test[:,0])
pred_all = model_all.predict(X_test)

RMSE(t_test, pred_single)
RMSE(t_test, pred_all)


x_single = t_test
y_single = pred_single
plt.title('First feature fit (CRIM)')
plt.xlabel('True House Prices')
plt.ylabel('Estimates')
plt.scatter(x_single,y_single)
#plt.legend()
plt.show()

x_all = t_test
y_all = pred_all
plt.title('Fitting on all features')
plt.xlabel('True House Prices')
plt.ylabel('Estimates')
plt.scatter(x_all,y_all)
#plt.legend()
plt.show()