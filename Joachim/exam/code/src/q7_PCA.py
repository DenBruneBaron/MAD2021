#!/usr/bin/env python
#
# MAD 2020-21, Exam number: 23
# Q7 (PCA)
#
# Code mainly developed or provided as part of A5

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from q7_clustering import centroids, assignments, data_norm as features

def __PCA(data):
    """
    From A5
    """
    data_cent = data - np.mean(data)
    Sigma = np.cov(data_cent.T)
    PCevals, PCevecs = np.linalg.eigh(Sigma)
    PCevals = np.flipud(PCevals) # vertical flip
    PCevecs = np.flip(PCevecs, axis=1) # horisontal flip
    return PCevals, PCevecs

def __transformData(features, PCevecs):
    """
    From A5
    """
    return np.dot(features,  PCevecs[:, 0:2])

PCevals, PCevecs = __PCA(features)

# Convert data to two dimemsions using PCA
features2D = __transformData(features, PCevecs)
centroids2D = __transformData(centroids, PCevecs)

def __visualizeLabels(features, centroids, referenceLabels):
    """
    From A5 (modified)
    """

    plt.figure()
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    y = referenceLabels

    plt.scatter(features[:, 0], features[:, 1], c = y, cmap = cmap_bold)
    plt.scatter(centroids[:, 0], centroids[:,1], c = 'black', s=100)
    plt.xlim(features[:, 0].min() - 0.1, features[:, 0].max() + 0.1)
    plt.ylim(features[:, 1].min() - 0.1, features[:, 1].max() + 0.1)
    plt.show()

__visualizeLabels(features2D, centroids2D, assignments)
