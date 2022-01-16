#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

mu = 2
sigma = 2
L = 1000
k = 1

# sampling from the proposal distribution
x = np.random.default_rng().normal(mu, sigma, L)

def unnormalized_distr(x):
    '''
    The unnormalized distribution ~p(x)
    '''
    return np.exp(-abs(x-2))

def proposal_distr(x):
    '''
    The proposal distribution q(x)
    '''
    return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2 * sigma**2))

print(proposal_distr(2))

k = unnormalized_distr(2) / proposal_distr(2)

print(k)
print(k * proposal_distr(10)-unnormalized_distr(10))

x = np.linspace(-10, 10, L)

print("k: %f" %np.mean(k))
plt.plot(x, unnormalized_distr(x), label='Unnormalised distribution')
plt.plot(x, proposal_distr(x), label='Proposal distribution')
plt.plot(x, k*stats.norm.pdf(x, mu, sigma), label='Proposal distribution stats')
plt.legend()
plt.xlabel('x')
plt.ylabel('Density')
plt.savefig('../plots/ex2_rejection_sampling.png')
plt.show()

# sample from q
