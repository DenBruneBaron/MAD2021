#!/usr/bin/env python

# MAD 2020-21, Assigment 6
# Newton-Raphson

import numpy as np
import matplotlib.pyplot as plt

import A6_Bayes as A6

##############################
# density functions from R&G #
##############################

def log_likelihood(t,X,w):
    '''
    The log-likelihood
    '''
    return np.sum(t * np.log(A6.sigmoid(X,w)) + (1-t) * np.log(1 - A6.sigmoid(X,w)))

def log_prior(w, sigma):
    '''
    The log-prior density
    '''
    D = w.shape[0];
    return -D/2 * np.log(2 * np.pi) - D * sigma - 1/(2 * sigma**2) * w.T * w

def log_posterior_unnorm(w, X, t, sigma):
    '''
    The unnormalized log-posterior
    '''
    return log_likelihood(t,X,w) + log_prior(w, sigma)

##################
# newton-raphson #
##################

def partial_derivative(w, X, t, sigma):
    '''
    The partial derivative of the unnormalized log-posterior with respect to the parameters
    '''
    return -1/sigma**2 * w + np.sum(X.T * (t + A6.sigmoid(X, w)))

def hessian(w, X, t, sigma):
    '''
    The Hessian matrix af partial derivatives
    '''
    dim = w.shape[0]
    return - 1/sigma**2 * np.identity(dim) - np.sum(
        X @ X.T * A6.sigmoid(X, w) * (1 - A6.sigmoid(X,w)))

def newton_raphson_step(w, X, t, sigma):
    '''
    The Newton-Rapshon method for updating parameters based on previous values
    '''
    return w - np.linalg.inv(hessian(w, X, t, sigma)) @ partial_derivative(w, X, t, sigma)

#########
# setup #
#########


# Test code
if (__name__=='__main__'):
    # load data
    t, X = A6.loaddata('../data/binary_classes.csv')

    print("X shape: ", X.shape)
    print((X @ X.T).shape)

    # initial parameter values
    w = np.array([0,0])

    # prior paramater
    sigma = np.sqrt(10)

    # number of itereations
    S = 10

    # print(log_likelihood(t,X,w))
    # print(log_prior(w,sigma))
    # print(log_posterior_unnorm(w,X,t,sigma))
    #print(A6.sigmoid(X,w))
    # print(partial_derivative(w,X,t,sigma))

    # print(newton_raphson_step(w, X,t,sigma))
    # print("sigmoid: ", A6.sigmoid(X,w))

    print(w)
    # # estimate
    for s in range(S):
        # print("UPDATE")
        # print("hessian: ", hessian(w,X,t,sigma))
        # print("hessian inv: ", np.linalg.inv(hessian(w, X, t, sigma)))
        # print("partial derivative: ", partial_derivative(w, X, t, sigma))
        w = newton_raphson_step(w, X, t, sigma)
        print(w)


    # #########
    # # plots #
    # #########
    A6.contourplot(t,X,w, A6.sigmoid)
    plt.title("Probability contours")
    # plt.show()
    plt.savefig('../plots/ex3_contours.png')
