import numpy as np

# class RidgeRegression(object):
#     """Linear least squares with l2 regularization.

#     This model solves a regression model where the loss function is
#     the linear least squares function and regularization is given by
#     the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
#     This estimator has built-in support for multi-variate regression
#     (i.e., when y is a 2d-array of shape [n_samples, n_targets]).

#     """
#     def __init__(self, alpha):
#         self.alpha = alpha

#     def fit(self, X, y):
#         """fit data
#         Args:
#             X: ndarray-data. training data with shape (n_samples, n_features)
#             y: ndarray-data. label with shape (n_samples, 1)

#         Returns:
#             coefs: w
#         """
#         n_samples, n_features = X.shape

#         # print(np.dot(np.transpose(X), X).shape)
#         w = np.dot(np.linalg.inv(np.dot(np.transpose(X), X) + self.alpha*np.eye(n_features, n_features)), np.dot(np.transpose(X), y))
#         return w

class RidgeRegression(object):
    
