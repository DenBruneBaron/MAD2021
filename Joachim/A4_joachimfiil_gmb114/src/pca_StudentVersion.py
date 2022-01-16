#!/usr/bin/env python
# coding: utf-8

# # Assignment: Principal Components Analysis (PCA)

# Task 1: Implement PCA on the diatoms database. Please output the proportion of variance explained by each of the first 10 components (5 points)
# 
# Task 2: Visualize fourth component of the PCA (3 points)
# 
# 
# 
# We start by loading the dataset found in the file 'diatoms.txt', which contains a set of *diatom* outlines. A diatom is a type of algae, whose species is strongly correlated with its outline shape; in the following, we will be using these outlines as a descriptive feature of the diatom.
# 
# The file 'diatoms.txt' contains 780 diatoms described by 90 successive "landmark points" (x_i, y_i) along the outline, recorded as (x_0, y_0, x_1, y_1, ..., x_89, y_89).
# 
# The file 'diatoms_classes.txt' contains one class assignment per diatom, into species classified by the integers 1-37.

# In[67]:


import numpy as np

diatoms = np.loadtxt('../data/diatoms.txt', delimiter=',').T
diatoms_classes = np.loadtxt('../data/diatoms_classes.txt', delimiter=',')
print('Shape of diatoms:', diatoms.shape)
print('Shape of diatoms_classes:', diatoms_classes.shape)
#print('Classes:', diatoms_classes)

d,N = diatoms.shape
print('Dimension:', d)
print('Sample size:', N)


# Here's a function that will plot a given diatom. Let's try it on the first diatom in the dataset.

# In[68]:


import matplotlib.pyplot as plt

def plot_diatom(diatom):
    xs = np.zeros(91)
    ys = np.zeros(91)
    for i in range(90):
        xs[i] = diatom[2*i]
        ys[i] = diatom[2*i+1]
    
    # Loop around to first landmark point to get a connected shape
    xs[90] = xs[0]
    ys[90] = ys[0]
    
    plt.plot(xs, ys)    
    plt.axis('equal')   

plot_diatom(diatoms[:,0])


# Let's next compute the mean diatom and plot it.

# In[69]:


mean_diatom = np.mean(diatoms, 1)
plot_diatom(mean_diatom)


# ### Task1: Implementing PCA
# 
# To implement PCA, please check the algorithm explaination from the lecture.
# Hits:
# 
# 1) Noramilize data subtracting the mean shape. No need to use Procrustes Analysis or other more complex types of normalization
# 
# 2) Compute covariance matrix (check np.cov)
# 
# 3) Compute eigenvectors and values (check np.linalg.eigh)

# In[92]:


import numpy.matlib

def pca(data):
    data_cent = data - np.mean(data, axis=0) # center data
    Sigma = np.cov(data_cent) # covariance matrix
    PCevals, PCevecs = np.linalg.eigh(Sigma) # eigenvalues and eigenvectors in ascending order
    PCevals = np.flipud(PCevals) # vertical flip  
    PCevecs = np.flip(PCevecs, axis=1) # horisontal flip
    
    return PCevals, PCevecs, data_cent

PCevals, PCevecs, data_cent = pca(diatoms)
# PCevals is a vector of eigenvalues in decreasing order. To verify, uncomment:
#print(PCevals)
# PCevecs is a matrix whose columns are the eigenvectors listed in the order of decreasing eigenvectors


# ***Recall:***
# * The eigenvalues represent the variance of the data projected to the corresponding eigenvectors. 
# * Thus, the 2D linear subspace with highest projected variance is spanned by the eigenvectors corresponding to the two largest eigenvalues.
# * We extract these eigenvectors and plot the data projected onto the corresponding space.

# ### Compute variance of the first 10 components
# 
# How many components you need to cover 90%, 95% and 99% of variantion. Submit the resulting numbers for grading.

# In[93]:


variance_explained_per_component = PCevals/np.sum(PCevals)
cumulative_variance_explained = np.cumsum(variance_explained_per_component)

plt.plot(cumulative_variance_explained)
plt.xlabel('Number of principal components included')
plt.ylabel('Proportion of variance explained')
plt.title('Proportion of variance explained as a function of number of PCs included')

# Let's print out the proportion of variance explained by the first 10 PCs
for i in range(10):
    print('Proportion of variance explained by the first '+str(i+1)+' principal components:', cumulative_variance_explained[i])


# ### Task2: Plot varianace associated with the fourth component
# 
# Please fill the gaps in the code to plot mean diatom shape with added FOURTH eigenvector mulitplied by [-3,-2,-1,0,1,2,3] standard deviations corresponding to this eigenvector.
# 
# Submit the resulting plot for grading.

# In[94]:


e4 = PCevecs[:, 3] # gets the fourth eigenvector
lambda4 = PCevals[3] # gets the fourth eigenvalue
std4 = np.sqrt(lambda4) # In case the naming std is confusing -- the eigenvalues have a statistical interpretation

diatoms_along_pc = np.zeros((7, 180))
for i in range(7):
    diatoms_along_pc[i] = mean_diatom + e4 * std4 *(i-3)
    
for i in range(7):
    plot_diatom(diatoms_along_pc[i])

plt.title('Diatom shape along PC4')



