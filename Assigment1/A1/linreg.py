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
        X = np.array(X).reshape((len(X), -1))
        #print("Reshaped array X")
        #print(X)
        t = np.array(t).reshape((len(t), 1))
        #print("Reshaped array t")
        #print(t)

        # prepend a column of ones
        ones = np.ones((X.shape[0], 1))
        #print("creating array of ones")
        #print(ones)
        X = np.concatenate((ones, X), axis=1)
        #print("concatenating original X array and array of ones")
        #print(X)
        # compute weights  (matrix inverse)
        self.w = np.linalg.pinv((np.dot(X.T, X)))
        self.w = np.dot(self.w, X.T)
        self.w = np.dot(self.w, t)
        print("Printing minimizer for linear regression (fit)")
        print(self.w)


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

        # # LØSNING 1 = Giver mig  w = [23.63506195 -0.43279318] OG prediction: [22361.4076994   -409.47067433]
        # transposed_matrix = X.T
        # vector_t = self.w
        # t_flat = vector_t.flatten()
        # mega_vector = np.full((253,2), t_flat)
        # #print(transposed_matrix)
        # #print(vector_t.flatten())
        # prediction = np.dot(transposed_matrix,mega_vector)
        # print("prediction:", prediction)

        # LØSNING 2 = Giver mig  fortsat en shape error

        X = np.array(X).reshape((len(X), -1))
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        #print(X.shape)
        #print(self.w.shape)
        #transposed_matrix = X.T
        #t_flat = self.w.flatten()
        prediction = np.dot(X,self.w)
        print("prediction:", prediction)
        return prediction
        #print(type(prediction))



