#!/usr/bin/env python
#
# MAD 2020-21, Exam number: 23
# Question 7 (Clustering)
#

import numpy as np


def normalize(M):
    """
    Normalize all columns in 2D Numpy-array M

    Returns the normalized M
    """
    return (M - np.mean(M, axis=0)) / np.std(M, axis=0)


def generate_seed_centroids(k, data, rng):
    """
    Randomly choose k datapoints from data matrix to be used as seed centroids

    Parameters:
    -----------
    k   : int number of centroids
    data   : (n_samples, n_features) datamatrix
    rng : np random generator

    Returns:
    --------
    centroids : Numpy array of k random centroids
    """
    centroids = np.empty((0,data.shape[1]))

    for _ in range(k):
        i = int(rng.random() * data.shape[0])
        datapoint = data[i]
        centroids = np.vstack((centroids, datapoint))

    return centroids

def dist(a,b):
    """
    Compute Euclidean distance between two points
    """
    return np.linalg.norm(a-b)

def get_nearest_centroid(datapoint, centroids):
    """
    Computes the index of the nearest centroid

    Params:
    ------
    datapoint : (n_features) np.array
    centroid  : (k, n_features) np.array of centroids

    Returns:
    -------
    index of neares centroid

    """
    distances = [dist(datapoint, centroid) for centroid in centroids]
    return np.argsort(distances)[0]

def assign_datapoints_to_centroids(data, centroids):
    """
    Assign datapoints to nearest centroids

    Params:
    -------
    data      : (n_samples, n_features) data matrix
    centroids : (k, n_features) array of centroids

    Returns:
    --------
    assignments : np.array of indices mapping each datapoint to
    its nearest centroid
    """
    assignments = [get_nearest_centroid(datapoint, centroids) for datapoint in data]
    return np.array(assignments)

def compute_sum_intra_cluster_dist(data, assignments, centroids):
    """
    Compute the sum of intra cluster distances of all k clusters

    Params:
    -------
    data          : (n_samples, n_features) data matrix
    assignments   : (n_samples) np.array
    centroids     : (k, n_features)


    Returns:
    -------
    computed intra cluster distance
    """
    k = len(centroids)

    # 3D-array of k clusters with assigned datapoints
    clusters = np.array([data[np.where(assignments == j)] for j in range(k)])

    sum = 0

    for j in range(k):
        sum += compute_intra_cluster_dist(clusters[j], centroids[j])

    return sum

def compute_intra_cluster_dist(cluster, centroid):
    """
    Compute the intra cluster distance of a single cluster
    """
    return np.sum([dist(datapoint, centroid) for datapoint in cluster])

def compute_new_centroids(data, assignments, centroids):
    """
    Compute new centroids

    Params:
    -------
    data          : (n_samples, n_features) data matrix
    assignments   : (n_samples) np.array
    centroids     : (k, n_features)

    Returns:
    --------
    centroids : (k, n_features) np.array
       The new centroids
    """
    k = len(centroids)

    # 3D-array of k clusters with assigned datapoints
    clusters = np.array([data[np.where(assignments == j)] for j in range(k)])

    for j in range(k):
        # number of datapoints in j'th cluster
        n = clusters[j].shape[0]
        if (n > 0):
            # update j'th centroid
            centroids[j] = 1/n * np.sum(clusters[j], axis=0)

    return centroids

def k_mean_clstr(k, data, centroids):
    """
    K-means clustering

    Params:
    ------
    k : number of clusters
    data : (n_samples, n_features) (normalized) data matrix
    rng  : random generator

    Returns:
    --------
    assignments, intra_cluster_dist : tuple

    """
    assignments = assign_datapoints_to_centroids(data, centroids)

    new_assignments = [] # initial dummy val

    while not np.array_equal(assignments, new_assignments):
        # repeat until assignments does not change
        centroids = compute_new_centroids(data, assignments, centroids)
        assignments = new_assignments
        new_assignments = assign_datapoints_to_centroids(data, centroids)

    intra_cluster_dist = compute_sum_intra_cluster_dist(data, assignments, centroids)

    return assignments, intra_cluster_dist


# load data
data = np.loadtxt('../data/seedsDataset.txt', delimiter=',')

print("Shape of data: %s" %str(data.shape))

# K-means clustering


# test
data_norm = normalize(data)

seed = 111
rng = np.random.default_rng(seed)

k = 3 # numbers of clusters

# running K-means clustering 5 times
# solution with smallest intra-cluster distance

res = np.empty((0,3))

for i in range(5):
    centroids = generate_seed_centroids(k,data_norm, rng)
    assignments, intra_cluster_dist = k_mean_clstr(k, data_norm, centroids)

    res = np.vstack((res, [centroids, assignments, intra_cluster_dist]))

# get results

centroids, assignments, intra_cluster_dist = res[np.argsort(res[:,2])[0]]

print("Smallest Intra-Cluster Distance: %.4f" %intra_cluster_dist)

# 3D-array of k clusters with assigned datapoints
clusters = np.array([data_norm[np.where(assignments == j)] for j in range(k)])

# print number of samples in each cluster
for j in range(k):
    print("Number of samples in cluster %d: %d" %(j, len(clusters[j])))
