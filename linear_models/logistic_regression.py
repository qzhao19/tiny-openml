import numpy as np
import numbers
from scipy import optimize
from utils import load_data


class LogisticRegressionClassifier(object):
    """Logistic Regression classifier
    
    Args:
        C : float, default=1.0
            Inverse of regularization strength; must be a positive float.
            Like in support vector machines, smaller values specify stronger
            regularization.
        
        solver : 
            Algorithm to use in the optimization problem.
        
    """
    def __init__(self, C=0.1, solver='bfgs'):
        self._C = C
        self._solver = solver
        
    def _sigmoid(self, X):
        """sigmoid function. Here, we must be careful of overflow in exp.
        
        f(x) = 1 / (1 +(exp(-x)))
        

        Parameters
        ----------
            X : ndarray 
                Input value.

        Returns
        -------
            The value of function sigmoid
        """
        # n_samples, n_features = np.shape(X)
        n_samples = len(X)
        h = np.zeros((n_samples, 1), dtype=float)
        # h = 1.0 / (1.0 + np.exp(-X))
        h = np.exp(np.fmin(X, 0)) / (1 + np.exp(-np.abs(X)))
        
        return h


    def _compute_cost_fn(self, theta, X, y):
        """Cost function 
        J(theat) = -1/m * {sum[y * log(h(x)) + (1-y) * log(1 - h(x))]}, h(x) = 1 / (1 - exp(-x))
        penalty = 1/m * (sum[theat * theta])

        Parameters
        ----------
            X : ndarray-like data
                input training data of shape (n_samples, n_features)
            y : ndarray_like
                label of target vector relative to X
            theta : float,
                 model weights
        Returns
        -------
            None.
        """
        n_sampels, n_features = np.shape(X)
        
        h = self._sigmoid(np.dot(X, theta))
        
        copied_theta = theta.copy()
        
        copied_theta[0] = 0
        
        penalty = (self._C / n_features) * np.dot(np.transpose(copied_theta), copied_theta)
        
        J_history = (-1 / n_features) * (np.dot(np.transpose(y), np.log(h)) + \
                                         np.dot(np.transpose(1 - y), np.log(1 - h))) 
        
        
        return J_history + penalty

    def _compute_gradients(self, theta, X, y):
        """Compute gradiend of the cost function
        
        theta_j = theta_j + alpha * (1/m) * Sum[(h(x_i) - y_i) * x_i]
        
        Parameters
        ----------
            X : ndarray-like data
                input training data of shape (n_samples, n_features)
            y : ndarray_like
                label of target vector relative to X
            theta : float,
                 model weights
        Returns
        -------
            None.
        """
        n_samples, n_features = np.shape(X)
        
        # define an array to storage gradients 
        grads = np.zeros(np.shape(theta)[0], dtype=float)
        
        # h_shape = n*p x p*1 = n*1
        h = self._sigmoid(np.dot(X, theta))
        copied_theta = theta.copy()
        
        copied_theta[0] = 0
        
        grads = (np.dot(np.transpose(X),(h - y)) / n_features) + \
                (self._C * copied_theta / n_features)
        
        return grads

    def _feature_mapping(self, x1, x2):
        """
        
        Parameters
        ----------
        x1 : TYPE
            DESCRIPTION.
        x2 : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        
        
        return 0
    
    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
            X : array_like of shape [n_samples, n_features]
                Training vector, where n_samples is the umber of samples
                and n_features is the number of features
            y : array_like of shape [n_samples, ]
                Target vector relative of traing data X

        Returns
        -------
            return gradients of the cost function == self._comtute_gradients
        """
        
        if (not isinstance(X, np.ndarray)) or (not isinstance(y, np.ndarray)):
            raise ValueError('Data or label must be array type')
        
        if not isinstance(self._C, numbers.Number) or self._C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)
        
        n_samples, n_features = X.shape
        
        
        theta = np.ones((n_features, 1), dtype=float)
        # penality_coef = self._C
        
        # J_history = self._compute_cost_fn(X, y, theta)
        
        if self._solver == 'bfgs':
            prob = optimize.fmin_bfgs(self._compute_cost_fn, \
                                      x0 = theta, \
                                      fprime = self._compute_gradients, \
                                      args = (X, y))
        
        return prob
        
    def predict(self, X, sample_weight):
        n_samples, n_features = X.shape
        
        pred = np.zeros((n_samples, 1), dtype=int)
        prob = self._sigmoid(np.dot(X, sample_weight))
        
        for i in range(n_samples):
            if prob[i] > 0.5:
                pred[i] = 1
            else:
                pred[i] = 0
                
        return pred


# if __name__ == "__main__":
#     path = './dataset/horse-colic-train.txt'
#     data, label = load_data(path)
    
#     print((data.shape))
#     print(label)
        
#     lr = LogisticRegressionClassifier()
#     prob = lr.fit(data, label)
    
#     print(prob)
        
        
        

