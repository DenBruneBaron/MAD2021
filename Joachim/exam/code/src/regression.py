#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def loaddata(filename):
    """Load the balloon data set from filename and return t, X
        t - N-dim. vector of target (temperature) values
        X - N-dim. vector containing the inputs (lift) x for each data point
    """
    # Load data set from CSV file
    Xt = np.loadtxt(filename, delimiter=',')

    # Split into data matrix and target vector
    X = Xt[:,0]
    t = Xt[:,1]

    return t, X


def predictiveplot(xnew, mu_pred, sigma2_pred, t, X):
    """Plots the mean of the predictive distribution (green curve) and +/- the predictive standard deviation (red curves).
        xnew - Mx1 vector of new input x values to evaluate the predictive distribution for
        mu_pred - Mx1 vector of predictive mean values evaluated at xnew,
        sigma2_pred - Mx1 vector of predictive standard deviation values evaluated at xnew
        t - vector containing the target values of the training data set
        X - vector containing the input values of the training data set
    """
    plt.figure()
    plt.scatter(X, t)
    plt.plot(xnew, mu_pred, 'g')
    plt.plot(xnew, mu_pred + np.sqrt(sigma2_pred).reshape((sigma2_pred.shape[0],1)), 'r')
    plt.plot(xnew, mu_pred - np.sqrt(sigma2_pred).reshape((sigma2_pred.shape[0],1)), 'r')



# Load data
t, X = loaddata('../data/hot-balloon-data.csv')


# Visualize the data
plt.figure()
plt.scatter(X, t)
plt.xlabel('Lift')
plt.ylabel('Temperature')
plt.title('Data set')



# This is a good range of input x values to use when visualizing the estimated models
xnew = np.arange(120, 300, dtype=np.float)

# # Exxample of how to use the predictiveplot function
# mu_fake = 0.25 * xnew.reshape((xnew.shape[0],1)) + 250.0
# sigma2_fake = mu_fake
# predictiveplot(xnew, mu_fake, sigma2_fake, t, X)
# plt.xlabel('Lift')
# plt.ylabel('Temperature')
# plt.title('Example of predictiveplot')



# ADD YOUR SOLUTION CODE HERE!

# import linear regression implementation provided in L3
import linreg

#
def augment(X, max_order):
    """ Augments a given data
    matrix by adding additional
    columns.

    NOTE: In case max_order is very large,
    numerical inaccuracies might occur ...
    """

    X_augmented = X

    for i in range(2, max_order+1):
        X_augmented = np.concatenate([X_augmented, X**i], axis=1)

    return X_augmented


def logarithmic(X):
    """Augments a given N x 1 data matrix by prepending
    a column of 1's and wrapping each x_n in a log-function.

    Returns the augmented N x 2 data matrix for the logarithmic model
    """
    # return np.concatenate([np.ones(len(X)).reshape((len(X), 1)),
    #                        np.log(X)], axis=1)
    return np.log(X)


# reshape both arrays to make sure that we deal with
# N-dimensional Numpy arrays
t = t.reshape((len(t),1))
X = X.reshape((len(X),1))

def optimal_sigma_squared(X,t,w_opt):
    """
    Maximum likelihood estimate for sigma_squared
    """

    # prepend a column of 1's
    X = np.concatenate([np.ones(len(X)).reshape((len(X), 1)),X], axis=1)

    #number of samples
    N = len(X)

    return 1/N * (t.T @ t - t.T @ X @ w_opt)

# polynomial model
K = 3
X_polynomial = augment(X, K)
polynomial_model = linreg.LinearRegression()
polynomial_model.fit(X_polynomial, t)
polynomial_model.sigma_squared = optimal_sigma_squared(X_polynomial,
                                                       t,
                                                       polynomial_model.w)

# logarithmic model
X_logarithmic = np.log(X)
logarithmic_model = linreg.LinearRegression()
logarithmic_model.fit(X_logarithmic, t)
logarithmic_model.sigma_squared = optimal_sigma_squared(X_logarithmic,
                                                        t,
                                                        logarithmic_model.w)

# printing
print("Polynomial model:")
print("Optimal w: \n %s" %str(polynomial_model.w))
print("Optimal sigma_squared:\n %s \n" %str(polynomial_model.sigma_squared))

print("Logarithmic model:")
print("Optimal w:\n %s" %str(logarithmic_model.w))
print("Optimal sigma_squared:\n %s\n" %str(logarithmic_model.sigma_squared))

# figures
# Code from Non_Linear_Regression.ipynb
Xplot = np.arange(X.min(), X.max(), 0.01)
Xplot = Xplot.reshape((len(Xplot),1))
Xplot = augment(Xplot, K)
pred_plot = polynomial_model.predict(Xplot)

polynomial_pred = polynomial_model.predict(X_polynomial)
logarithmic_pred = logarithmic_model.predict(X_logarithmic)

plt.plot(X, polynomial_pred, 'x', color='orange')
plt.plot(X, logarithmic_pred, 'x', color='purple')
plt.plot(Xplot[:,0], pred_plot, '-', color='red', label="Polynomial Model")

Xplot = np.arange(X.min(), X.max(), 0.01)
Xplot = Xplot.reshape((len(Xplot),1))
pred_log_plot = logarithmic_model.predict(np.log(Xplot))

plt.plot(Xplot[:,0], pred_log_plot, '-', color='green', label="Logarithmic Model")
plt.legend()

#
# Bayesian regression
#
# Code developed in A4

def Sigmaw(X, variance, Sigma_0):
    '''
    Compute the posterior Sigma_w
    '''

    # prepend a column of 1's
    X = np.concatenate([np.ones(len(X)).reshape((len(X), 1)),X], axis=1)

    return np.linalg.inv(1/variance * X.T @ X + np.linalg.inv(Sigma_0))

def muw(X, t, variance, Sigma_0, mu_0, Sigma_w):
    '''
    Compute the posterior mu_w
    '''

    # prepend a column of 1's
    X = np.concatenate([np.ones(len(X)).reshape((len(X), 1)),X], axis=1)

    return Sigma_w @ (1/variance * X.T @ t + np.linalg.inv(Sigma_0) @ mu_0)

# parameters used for both models
variance = 25 # likelihood variance
sigma_squared_0 = 10

# polynomial_model
mu_0_polynomial = np.array([268,0,0,0]).reshape((4,1))
Sigma_0_polynomial = np.eye(K+1) * sigma_squared_0
Sigma_w_polynomial = Sigmaw(X_polynomial, variance, Sigma_0_polynomial)
mu_w_polynomial = muw(X_polynomial, t, variance, Sigma_0_polynomial, mu_0_polynomial,
                      Sigma_w_polynomial)

xnew = xnew.reshape((len(xnew),1))
xnew = augment(xnew, K)
xnew = np.concatenate([np.ones(len(xnew)).reshape((len(xnew), 1)), xnew], axis=1)

mu_pred = xnew @ mu_w_polynomial

sigma2_pred = np.diag(variance + xnew @ Sigma_w_polynomial @ xnew.T)
sigma2_pred = sigma2_pred.reshape((len(sigma2_pred), 1))

predictiveplot(xnew[:,1], mu_pred, sigma2_pred, t, X)
plt.xlabel('Lift')
plt.ylabel('Temperature')
plt.title('Predictive plot for the polynomial model')

# logarithmic model
mu_0_logarithmic = np.array([133,32]).reshape((2,1))
Sigma_0_logarithmic = np.eye(2) * sigma_squared_0
Sigma_w_logarithmic = Sigmaw(X_logarithmic, variance, Sigma_0_logarithmic)
mu_w_logarithmic = muw(X_logarithmic, t, variance, Sigma_0_logarithmic,
                       mu_0_logarithmic,
                       Sigma_w_logarithmic)

xnew = xnew[:,1].reshape((len(xnew[:,1]),1))
xnew_log = np.log(xnew)
xnew_log = np.concatenate([np.ones(len(xnew_log)).reshape(
    (len(xnew_log), 1)), xnew_log], axis=1)

mu_pred = xnew_log @ mu_w_logarithmic

sigma2_pred = np.diag(variance + xnew_log @ Sigma_w_logarithmic @ xnew_log.T)
sigma2_pred = sigma2_pred.reshape((len(sigma2_pred), 1))

predictiveplot(xnew, mu_pred, sigma2_pred, t, X)
plt.xlabel('Lift')
plt.ylabel('Temperature')
plt.title('Predictive plot for the logarithmic model')


# Show all figures
plt.show()
