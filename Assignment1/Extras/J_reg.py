import numpy as np

# NOTE: This template makes use of Python classes. If
# you are not yet familiar with this concept, you can
# find a short introduction here:
# http://introtopython.org/classes.html

class LinearRegression():
    """
    Linear regression implementation.
    """

    def __init__(self):

        pass

    def fit(self, X, t):
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        """
# make sure that we have Np arrays; also
# reshape the target array to ensure that we have
# a N-dimensional Np array (ndarray), see
# https://docs.scipy.org/doc/np-1.13.0/reference/arrays.ndarray.html
        n_samples = X.shape[0]
        X = np.array(X).reshape((n_samples, -1))
        t = np.array(t).reshape((n_samples, 1))

        # prepend a column of ones
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        
        # compute weights (solve system)
        a = np.dot(X.T, X)
        b = np.dot(X.T, t)

        self.w = np.linalg.solve(a,b)


    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """

    # make sure that we have Np arrays; also
    # reshape the array to ensure that we have
    # a N-dimensional Np array (ndarray), see
    # https://docs.scipy.org/doc/np-1.13.0/reference/arrays.ndarray.html
        X = np.array(X).reshape((len(X), -1))

        # prepend a column of ones
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

        # compute predictions
        predictions = np.dot(X, self.w)

        return predictions