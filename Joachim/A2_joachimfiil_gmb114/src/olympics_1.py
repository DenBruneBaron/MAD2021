'''
Assigment 2
Exercise 2.a

Polynomial Fitting with Regularized Linear Regression and Cross-Validation

Fitting a first order polynomial

Joachim Fiil
gmb114@alumni.ku.dk

'''

import numpy as np
import linreg # with regularization
import matplotlib.pyplot as plt

# load data
data = np.loadtxt('../data/men-olympics-100.txt', delimiter=' ')

# LOOCV with regularized linear regression and lambda in range [0;1]
K = np.arange(len(data))

lambda_range = np.logspace(-8,0,100, base=10)
lambda_range = np.insert(lambda_range, 0, .0) # prepend lamba = 0

loss_per_lambda = []
coefficients_per_lamba = []

results = np.empty(1)

for lam in lambda_range:
    errors_train = []
    errors_validation = []
    coefficients = []

    for i in K:
        # partition data
        data_val = data[i:i+1,:]
        
        if i == 0:
            data_train = data[i+1:,:]
        elif i == K.max():
            data_train = data[:i,:]
        else:
            data_train = np.concatenate((data[:i,:], data[i+1:,:]))
            
        # validation set
        t_val = data_val[:,1:2] # first place running times (second column)
        X_val = data_val[:,0:1] # year (first column)
        
        # training set
        t_train = data_train[:,1:2]
        X_train = data_train[:,0:1]
        
        # fit model on training set
        model = linreg.LinearRegression(lam)
        model.fit(X_train, t_train)
        coefficients.append(model.w)
        
        # get training predictions and error
        preds_train = model.predict(X_train)
        error_train = ((preds_train - t_train)**2).mean()
        
        # get validation prediction and error
        preds_val = model.predict(X_val)
        error_val = ((preds_val - t_val)**2).mean()
        
        errors_train.append(error_train)
        errors_validation.append(error_val)

    # average squared validation loss for
    loss = np.array(errors_validation).mean()
    loss_per_lambda.append(loss)

    coefficients_per_lamba.append(coefficients)

    
    # results = np.append(results, np.array([lam, loss, np.array(coefficients)]))
    # print(np.array([lam, loss, coefficients]))
    print("lam=%.10f and loss=%.10f" % (lam, loss))

# finding minimum loss and corresponding best lambda 
loss_per_lambda = np.array(loss_per_lambda)
min_loss = loss_per_lambda.min()
best_index = np.where(loss_per_lambda == min_loss)[0][0]
print("index: ", best_index)

print("\tBest lambda: %.10f" % lambda_range[best_index]) 
print("\tMinimum loss: %.10f" % min_loss)
print("")
print("\tSet of coefficients for best lambda:")

for coeff in coefficients_per_lamba[best_index]:
    print("\t\tw0: %.10f, w1: %.10f" % (coeff[0], coeff[1]))


print("\n")
print("\tSet of coefficients for lamba = 0:\n")

for coeff in coefficients_per_lamba[0]:
    print("\t\tw0: %.10f, w1: %.10f" % (coeff[0], coeff[1]))
    
# plotting the LOOCV error as a function of lambda
plt.plot(lambda_range, loss_per_lambda)
plt.xticks(lambda_range)
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Average squared validation loss for LOOCV")
plt.title("First order polynomial")
#plt.savefig("../plots/ex2_a.png")
plt.show()
    
    
    
