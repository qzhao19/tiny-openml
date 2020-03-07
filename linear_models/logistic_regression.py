import numpy as np
# import matplotlib.pyplot as plt

# class LogisticRegressionClassifier(object):
#     """Logistic Regression classifier.
#     """
#     def __init__(self):
#         pass

#     def _sigmoid(self, x):
#         """Compute sigmoid function. Here, we must be careful of overflow in exp
#         Args:
#             x: float, input value
#         """
#         return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))

#     def _compute_gradient_descent(self, data, label, alpha, max_steps):
#         """Compute gradient ascent to optimize max vals of function

#         Args:
#             data: ndarray-like data, input data
#             label: ndarray_like, label
#             alpha: float, learning rate when update weights
#             max_steps: int, max iterations to update weights

#         """
#         if (not isinstance(data, np.ndarray)) or (not isinstance(label, np.ndarray)):
#             raise ValueError('Data or label shoule be array type')
        
#         # get the number of sample to define ths shape of weights and biases
#         n_samples, n_features = data.shape
#         # init weights as 1
#         weights = np.ones((n_features, 1), dtype=data.dtype)

#         label = label.reshape((n_samples, 1))
#         for step in range(max_steps):
#             # forcast predict vals from updated weighted data
#             sigmoid = self._sigmoid(np.dot(data, weights))
#             error = label - sigmoid
#             # update weights, if is '-', it will compute gradient descent 
#             weights = weights + alpha * np.transpose(data).dot(error)
#         return weights

#     def fit(self, train_X, train_y, alpha=1e-3, max_steps=500):
#         return self._compute_gradient_descent(train_X, train_y, alpha, max_steps)

#     def predict(self, test_X, test_y, weights):
#         """Predict test data
#         """
#         if (not isinstance(test_X, np.ndarray)) or (not isinstance(test_y, np.ndarray)):
#             raise ValueError('Data or label shoule be array type')
        
#         # get the number of sample to define ths shape of weights and biases
#         n_samples, n_features = test_X.shape
#         sigmoid = self._sigmoid(np.dot(test_X, weights))
#         error = 0.0
#         label = test_y

#         for i in range(n_samples):
#             if sigmoid[i] > 0.5:
#                 print(str(i+1)+'-th sample ', int(label[i]), 'is classfied as: 1') 
#                 if label[i] != 1:
#                     error += 1
#             else:
#                 print(str(i+1)+'-th sample ', int(label[i]), 'is classfied as: 0')
#                 if label[i] != 0:
#                     error += 1
#         error_rate = error/n_samples
#         print ("error rate is:", "%.4f" %error_rate)
#         return error_rate


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
    def __init__(self, C=1.0, solver='bfgs'):
        self._C = C
        self._solver = solver
        
    def _sigmoid(self, x):
        """sigmoid function. Here, we must be careful of overflow in exp.
        
        f(x) = 1 / (1 +(exp(-x)))
        

        Parameters
        ----------
            x : float
                Input value.

        Returns
        -------
            The value of function sigmoid
        """
        return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))
    
    def _compute_cost_fn(self, X, y, theta, max_steps=100):
        """Cost function 
        J(theat) = -1/m * {sum[y * log(h(x)) + (1-y) * log(1 - h(x))]}, h(x) = 1 / (1 - exp(-x))
        

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

    def _compute_gradient(self, X, y, theta):
        """Compute gradiend of the cost function
        
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





