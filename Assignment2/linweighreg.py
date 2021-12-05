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
        # TODO: YOUR CODE HERE
        n = X.shape[0]
        X = np.array(X).reshape((n, -1))
        t = np.array(t).reshape((n, 1))


        # prepend a column of ones
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

        A = np.identity(len(t))
        A_pow = A*(t**2)
        #print("Diagonal M: ",A_pow.shape)
        #print("t shape : ",t.shape)
        #print(A_pow)


        # compute weights (solve system)
        a = np.dot(X.T, A_pow)
        #print("a shape:", a.shape)
        b = np.dot(a, X)
        #print("b shape:", b.shape)
        c = np.dot(a,t)
        #print("c shape:", c.shape)

        self.w = np.linalg.solve(b,c)

        #return(self.w)



#---------------------------------------------------
    def fit_LOOCV(self, X, t, lamda, N):
        # Create identity matrix
        n = X.shape[0]
        X = np.array(X).reshape((n, -1))
        t = np.array(t).reshape((n, 1))
        idm = np.identity(len(X))
        
        # X^T * X
        a1 = np.dot(X.T, X)
        print("a shape:", a1.shape)
        # X^T * t
        b2 = np.dot(X.T, t)
        print("b shape:", b2.shape)
        # N * Lamda * identity_matrix
        INLamda = N * idm
        #print("INL shape:", INLamda)
        #print(INLamda)
        # First block of equation (X^T * X + N * Lamda * identity_matrix)
        #fst_block = a * INLamda

        #self.w = np.linalg.solve(fst_block, b)

#--------------------------------------------------

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

        # TODO: YOUR CODE HERE
        X = np.array(X).reshape((len(X), -1))

        # prepend a column of ones
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

        # compute predictions
        prediction = np.dot(X, self.w)

        return prediction




