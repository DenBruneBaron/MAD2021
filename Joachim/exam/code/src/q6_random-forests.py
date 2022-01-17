#!/usr/bin/env python
#
# MAD 2020-21, Exam
# Question 6, Classification & Random Forests
#

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

# load data
data_train = np.loadtxt(
    '../data/accent-mfcc-data_shuffled_train.txt',
    delimiter=',')
data_validation = np.loadtxt(
    '../data/accent-mfcc-data_shuffled_validation.txt',
    delimiter=',')

t_train = data_train[:,0]
X_train = data_train[:,1:]

t_validation = data_validation[:,0]
X_validation = data_validation[:,1:]

print("Shape of training data: %s" %str(data_train.shape))
print("Shape of validation data: %s"%str(data_validation.shape))
print("Shape of training targets: %s" %str(t_train.shape))
print("Shape of training features: %s" %str(X_train.shape))
print("Shape of validation targets: %s" %str(t_validation.shape))
print("Shape of validation features: %s" %str(X_validation.shape))
print()

# b) finding optimal set of random forest classifier parameters
param_grid = {
    'criterion'    : ['gini', 'entropy'],
    'max_depth'    : [2,5,7,10,15],
    'max_features' : ['sqrt', 'log2']
    }

res = np.empty((0,3)) # for resulting metrics

for params in list(ParameterGrid(param_grid)):
    # setup classifier using parameters
    clf = RandomForestClassifier(
        criterion    = params['criterion'],
        max_depth    = params['max_depth'],
        max_features = params['max_features'])

    # train
    clf.fit(X_train, t_train)

    # number of correctly classified validation samples
    t_pred = clf.predict(X_validation)
    acc_score = accuracy_score(t_validation, t_pred)

    # probability associated with classification
    t_prob = clf.predict_proba(X_validation)
    prob_score = np.mean([t_prob[int(t_val)]
                          for (t_prob, t_val)
                          in zip(t_prob, t_validation)])

    print("Accuracy score: %.2f"
        %acc_score)
    print("Average probability assigned to correct classes: %.2f"
        %prob_score)

    # print params if more optimal than previously tried
    if len(res) > 0 and (acc_score > res[-1,1]
                         or (acc_score == res[-1,1]
                             and prob_score > res[-1,2])):
        print(params)

    # accumulate results
    res = np.append(res, np.array([[params, acc_score, prob_score]]), axis=0)
    res = res[np.lexsort((res[:,1], res[:,2]))] # sort ascending

for t in res:
    print(t[0])
