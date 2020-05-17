# from utils import load_data
import numpy as np
from utils import load_data



class LinearRegression(object):
    """LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters:
        normalize : bool, optional, default False. If True, the regressors X 
            will be normalized before regression by subtracting the mean and 
            dividing by the l2-norm
        
        solver: string, default value is {closed_form, sgd}
                Solver to use in the computational routines,
                (决定了我们对回归损失函数的优化方法) 
        
    
    
    """
    def __init__(self, normalize, solver=None):
        self.normalize = normalize
        self.solver = solver
    
    def _normalize_data(self, X):
        """normalize data by substracting the mean and dividing sigma
        """
        
        n_samples, n_features = X.shape
        x_norm = np.zeros((n_samples, n_features), dtype=X.dtype)

        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)

        for i in range(n_features):
            x_norm[:, i] = (X[:, i] - mu[i]) / sigma[i]

        return x_norm

    
    def _fit_closed_form(self, X, y=None):
        """Using closed form solution to fit model, 
        
                delta = (X.T * X)^(-1) * X.T * y
        

        Parameters
        ----------
            X : ndarray of shape [n_samples, n_featiures]
                Training data.
            y : ndarray of shape (n_samples, ), optional
                Target values. The default is None.

        Returns
        -------
            None.

        """
        # get the transpose of matrix X and matrix product between X and X_T
        X_T = np.transpose(X)
        X_product = np.matmul(X, X_T)
        
        try:
            X_inv = np.linalg.inv(X_product)
        except Exception as error:
            raise error
        
        theta = np.matmul(np.matmul(X_inv, X_T), y)

        self.theta = theta
        
    def _fit_sgd(self, X, y=None, alpha=0.1, n_iters=100):
        """Gradient descend method to fit model 
        
    
        Parameters
        ----------
            X: ndarray-like data of shape [n_samples, n_features], 
                input training data
            y: ndarray_like data of shape [n_samples,]
                Target label
            alpha: float, 
                learning rate when update weights
            max_steps: int, max iterations to update weight

        Returns
        -------
        None.

        """
    
        n_samples, n_features = X.shape
        
        theta = np.zeros((n_features + 1, 1), np.float32)
        
        cost_func = 










# class LinearRegression(object):
#     """LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
#     to minimize the residual sum of squares between the observed targets in
#     the dataset, and the targets predicted by the linear approximation.

#     Args:
#         normalize : bool, optional, default False. If True, the regressors X 
#         will be normalized before regression by subtracting the mean and 
#         dividing by the l2-norm

#     """
#     def __init__(self, standard=False):
#         self._standard = standard

#     def _standardize_data(self, X):
#         """normalize data by substracting the mean and dividing sigma
#         """
#         if not isinstance(X, np.ndarray):
#             raise ValueError('Input data is not ndarray type')
        
#         n_samples, n_features = X.shape
#         x_norm = np.zeros((n_samples, n_features), dtype=X.dtype)

#         mu = np.mean(X, axis=0)
#         sigma = np.std(X, axis=0)

#         for i in range(n_features):
#             x_norm[:, i] = (X[:, i] - mu[i])/sigma[i]

#         return x_norm
    
#     def _compute_cost(self, X, y, theta):
#         """compute cost function, e.g. f(x) = (1/N)*{(wx-y)'(wx_y)}
#         """
#         n_samples = len(y)
#         return (np.transpose(X * theta - y)) * (X * theta - y)/(2 * n_samples)
    
#     def _compute_gradient(self, X, y, theta, alpha, n_iters):
#         """Compute gradient descent to optimize vals of function
#         Args:
#             X: ndarray-like data, input data
#             y: ndarray_like, label
#             theta: model weights
#             alpha: float, learning rate when update weights
#             max_steps: int, max iterations to update weight
        
#         Returns:
#             model weights theta and cost of object function J_history

#         """
#         n_samples = len(y)
#         n_weights = len(theta)

#         # temp variabel to keeping theta of each iteration computation
#         temps = np.matrix(np.zeros((n_weights, n_iters), dtype=np.float))
#         J_history = np.zeros((n_iters, 1), dtype=np.float)

#         for i in range(n_iters):
#             # compute the product of X and weights
#             h = np.dot(X, theta)
#             # compute gradients
#             temps[:, i] = theta - (alpha / n_samples) * (np.dot(np.transpose(X), h - y))
#             theta = temps[:, i]
#             # call object cost function 
#             J_history = self._compute_cost(X, y, theta)
#         return theta, J_history 
    
#     def fit(self, X, y, alpha, n_iters=100):
#         """
#         """
        
#         n_samples, n_features = X.shape
        
#         if self._standard:
#             X = self._standardize_data(X)
            
#         X = np.hstack((np.ones((n_samples, 1), dtype=float), X))
        
#         theta = np.zeros((n_features + 1, 1), dtype=float)
        
#         y = y.reshape(-1,1)
        
#         theta,J_history = self._compute_gradient(X, y, theta, alpha, n_iters)
        
#         return theta
            
#     def predict(self, X, sample_weight):
#         if self._standard:
#             X = self._standardize_data(X)
            
#         n_samples, n_features = X.shape
        
#         X = np.hstack((np.ones((n_samples, 1), dtype=float), X))

#         return np.dot(X, sample_weight)
        
    
#     def score(self, y_true, y_pred, sample_weight):
#         """
#         Return the coefficient of determination R^2 of the prediction. 
#         The coefficient R^2 is defined as (1 - u/v), where u is the 
#         residual sum of squares ((y_true - y_pred) ** 2).sum() and 
#         v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). 
#         The best possible score is 1.0 and it can be negative (because 
#         the model can be arbitrarily worse).

#         Parameters
#         ----------
#             y_true : float
#                 true value
#             y_pred : float
#                 prediected value from model
#             sample_weight : float
#                 sample weight of training data.

#         Returns
#         -------
#             R^2 of self.predict(X) wrt. y..
#         """
#         u = ((y_true - y_pred) ** 2).sum()
#         v = ((y_true - y_true.mean()) ** 2).sum()
        
#         return 1- u/v      
        
    
#     def test(self, X, y, sample_weight):
        
#         # return self._standardize_data(X)
#         return self.predict(X, sample_weight)
#         # return self.fit(X, y, alpha=0.01)
        
        
# if __name__ == "__main__":
#     path = './dataset/abalone.txt'
#     data, label = load_data(path)
    
#     print(len(data[:4000, :]))
#     print(len(label[:4000, :]))
    
#     print(data)
    
#     lr = LinearRegression()
    
#     coefs = lr.fit(data[:4000, :], label[:4000], 0.01)
    
#     result = lr.test(data[4001:, :], label[4001:, :], coefs)
    
#     print(coefs)
    
#     # print(result)






