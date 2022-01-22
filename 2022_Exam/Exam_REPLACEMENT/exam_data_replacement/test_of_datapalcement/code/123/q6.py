# Modelling and Analysis of Data
# Exam 2022 : Date  17th - 25th of January
# Exam no: 39

# Question 6 (Classification & Validation, 4 points)
# Question A and B


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier

# Modified the loaddata function to make the training features into
# 2d np.array, more specifically the 't value'
def loaddata(filename):
    """Load the accent-mfcc-data set from filename and return t, X
        t - N-dim. vector of target (temperature) values
        X - N-dim. vector containing the inputs (lift) x for each data point
    """
    # Load data set from CSV file
    Xt = np.loadtxt(filename, delimiter=",")
    
    # Split into data matrix and target vector
    X = Xt[:,0] # Matrix
    t = Xt[:,1:] # Vector
    
    return t, X

'''
QUESTION 6 - A
'''
# Loads the training data
X_train, t_train = loaddata("./accent-mfcc-data_shuffled_train.txt")

# Loads the validation data
X_validation, t_validation = loaddata("./accent-mfcc-data_shuffled_validation.txt")


print("Shape of training targets: %s" %str(t_train.shape))
print("Shape of training features: %s" %str(X_train.shape))
print("Shape of validation targets: %s" %str(t_validation.shape))
print("Shape of validation features: %s" %str(X_validation.shape))
print()


'''
QUESTION 6 - B
'''
# Adding the parameters to test given in the exam question.
# Using these to find the optimal set of random forest classifier parameters.
random_f_parameters = {
    'criterion'         : ['gini', 'entropy'],
    'max_tree_depth'    : [2,5,6,10,15],
    'max_features'      : ['sqrt', 'log2']
    }

# Empty array for the result metrics
res = np.empty((0,3)) 

# Looping through the chosen(given) parameters and setting up the
# Random forest classifier each time.
for params in list(ParameterGrid(random_f_parameters)):
    clf = RandomForestClassifier(
        criterion    = params['criterion'],
        max_depth    = params['max_tree_depth'],
        max_features = params['max_features'])

    # Training using the created classifier with the given parameters.
    clf.fit(X_train, t_train)

    # number of correctly classified validation samples
    t_prediction = clf.predict(X_validation)
    acc_score = accuracy_score(t_validation, t_prediction)

    # probability associated with classification
    t_probability = clf.predict_proba(X_validation)
    probability_score = np.mean([t_probability[int(t_val)]
                          for (t_probability, t_val)
                          in zip(t_probability, t_validation)])

    print("Accuracy score: %.2f"
        %acc_score)
    print("Average probability assigned to correct classes: %.2f"
        %probability_score)

    # print the parameters if new ones are more optimal 
    # than previously tried ones.
    if len(res) > 0 and (acc_score > res[-1,1]
                         or (acc_score == res[-1,1]
                             and probability_score > res[-1,2])):
        print(params)

    # accumulate results
    res = np.append(res, np.array([[params, acc_score, probability_score]]), axis=0)
    # sort the results in ascending order
    res = res[np.lexsort((res[:,1], res[:,2]))] 

for x in res:
    print(x[0])
