#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

mu, sigma = 2, 2 #

def density(x, mu, sigma):
    '''
    Univariate Gaussian density to be used when computing the entropy H(X)
    '''
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2 / (2*sigma**2))



# initialise random generator
rng = np.random.default_rng()


def mean(rng, mu, sigma, N_samples):
    '''
    Use Monte Carlo Integration to approximate mean E(X)
    '''
    return np.mean(rng.normal(mu, sigma, N_samples))

def kurtosis(rng , mu, sigma, N_samples):
    '''
    Use Monte Carlo Integration to approximate kurtosis Kurt(X)
    '''
    return np.mean(((rng.normal(mu, sigma, N_samples)-mu)/sigma)**4)-3

def entropy(s):
    '''
    Use Monte Carlo Integration to approximate entropy H(X)
    '''
    return np.mean(-np.log(density(s, mu, sigma)))

#mean = np.mean(s)
#kurtosis = np.mean(((s-mu)/sigma)**4)-3
#entropy = np.mean(-np.log(density(s, mu, sigma)))

print("Expectation: %f" %mean(rng, mu, sigma, 1000))
# print("Kurtosis: %f" %kurtosis(s))
# print("Entropy: %f" %entropy(s))
print("Entropy (analytical): %f" %(np.log(sigma*np.sqrt(2*np.pi*np.exp(1)))))

# plotting the approximate values for the three expectation functions in same figure
plt.figure()

N_range = np.arange(1000,1010)

print("N_range.shape: ", N_range.shape)

# range of number of samples used in approximations
for N in N_range:
    print(mean(rng, mu,sigma, N))


# mean E(X)
plt.subplot(131)
# plt.plot(N, mean(rng, mu, sigma, N))
plt.title("Mean E(X)")

# kurtosis Kurt(X)
plt.subplot(132)
plt.title("Kurtosis Kurt(X)")

# entropy H(X)
plt.subplot(133)
plt.title("Entropy H(X)")

# show all figures
plt.suptitle("Approximations")
plt.show()
