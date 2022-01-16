#!/usr/bin/env python

# MAD 2020-21, Assignment 6
# Joachim Fiil (gmb114)

# Implementation of Metropolis-Hastings method

import numpy as np
import matplotlib.pyplot as plt

import A6_Bayes as A6

# Load the data
t, X = A6.loaddata('../data/binary_classes.csv')

# proposal density variance rho**2 = 0.5
rho_sqr = 0.5

w_start = np.array([2.2,2.5])
prop_cov = rho_sqr * np.identity(2)
print(prop_cov)

print(prop_cov * w_start.T)

print(w_start)
print(w_start.shape)

w = np.random.default_rng().multivariate_normal(w_start, prop_cov, 1)
print(w.reshape(w_start.shape[0]))


def generate_proposal(w_old, rng):
    '''
    Generates a new proposal sample by sampling from a Gaussian distribution
    centered at w_old
    '''
    return rng.multivariate_normal(w_old, cov, 1).reshape(w_old.shape[0])

def acceptance_ratio():
    '''
    The ration of posteriors multiplied by the ratio of likelihoods
    --Or the ratio of unnormalized posteriors
    '''
