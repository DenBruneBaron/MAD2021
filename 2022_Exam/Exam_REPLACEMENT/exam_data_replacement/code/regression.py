import numpy as np
import matplotlib.pyplot as plt


def loaddata(filename):
    """Load the balloon data set from filename and return t, X
        t - N-dim. vector of target (temperature) values
        X - N-dim. vector containing the inputs (lift) x for each data point
    """
    # Load data set from CSV file
    Xt = np.loadtxt(filename, delimiter=',')

    # Split into data matrix and target vector
    X = Xt[:,0]
    t = Xt[:,1]
    
    return t, X
    
    
def predictiveplot(xnew, mu_pred, sigma2_pred, t, X):
    """Plots the mean of the predictive distribution (green curve) and +/- the predictive standard deviation (red curves).
        xnew - Mx1 vector of new input x values to evaluate the predictive distribution for
        mu_pred - Mx1 vector of predictive mean values evaluated at xnew,
        sigma2_pred - Mx1 vector of predictive standard deviation values evaluated at xnew 
        t - vector containing the target values of the training data set 
        X - vector containing the input values of the training data set
    """
    plt.figure()
    plt.scatter(X, t)
    plt.plot(xnew, mu_pred, 'g')
    plt.plot(xnew, mu_pred + np.sqrt(sigma2_pred).reshape((sigma2_pred.shape[0],1)), 'r')
    plt.plot(xnew, mu_pred - np.sqrt(sigma2_pred).reshape((sigma2_pred.shape[0],1)), 'r')



# Load data
t, X = loaddata('./data/accent-mfcc-data_shuffled_train.txt')


# Visualize the data
plt.figure()
plt.scatter(X, t)
plt.xlabel('Lift')
plt.ylabel('Temperature')
plt.title('Data set')



# This is a good range of input x values to use when visualizing the estimated models
xnew = np.arange(120, 300, dtype=np.float)

# Exxample of how to use the predictiveplot function
mu_fake = 0.25 * xnew.reshape((xnew.shape[0],1)) + 250.0
sigma2_fake = mu_fake
predictiveplot(xnew, mu_fake, sigma2_fake, t, X)
plt.xlabel('Lift')
plt.ylabel('Temperature')
plt.title('Example of predictiveplot')



# ADD YOUR SOLUTION CODE HERE!



# Show all figures
plt.show()

