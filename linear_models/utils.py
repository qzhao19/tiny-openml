# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:22:19 2020

@author: qizhao
"""
import numpy as np
import matplotlib.pyplot as plt

# def load_data(path):
#     """load dataset
#     Args:
#         path: txt file path
#     """
#     data, label = [], []
#     with open(path, 'r') as file:
#         num_features = len(file.readline().split('\t')) - 1
#         lines = [line.strip().split('\t') for line in file.readlines()]
#     data, label = [], []
#     for i in range(len(lines)):
#         data.append(lines[i][0:num_features])
#         label.append(lines[i][-1])
    
#     return np.array(data, dtype=np.float), np.array(label, dtype=np.float)


def load_data(file_path, sep = '\t'):
    """Download dating data to testing alogrithm

    Args:
        file_name: input file path which allows to read a txt file
    Returns:
        retuen_mat: a matrix of dating data contains 3 attributs: 
                        1. Number of frequent flyer miles earned per year
                        2. Percentage of time spent playing video games
                        3. Liters of ice cream consumed per week
        label_vect: a vectro conatins labels 
    """
    data = []
    label = []
    with open(file_path, 'r') as file:
        contents = [line.strip().split(sep) for line in file.readlines()]

    for i in range(len(contents)):
        data.append(contents[i][:-1])
        label.append(contents[i][-1])

    return np.array(data, dtype=float), np.array(label, dtype=float)



# def linear_regression(X, y):
#     """standard linear regression function
#     Args:
#         X: float, dataset
#         y: float, label
#     """
#     # make sur that matrix shoule be full rank, which means X.T * X inversible
#     if np.linalg.det(np.dot(np.transpose(X), X)) == 0:
#         raise ValueError('This matrix is singular, cannot do inverse')
#     w = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), (np.dot(np.transpose(X), y)))
#     return w

def plot_dataset(X, y):
    """plot Scatter figure
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 1], y, s=20, c='blue', alpha=0.5)
    plt.title('DataSet')
    plt.xlabel("X")
    plt.xlabel("y")
    plt.show()
    
def plot_regression(X, y, ws):
    """plot linear regression equation
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X[:,1], np.dot(X, ws), c = 'red') 
    ax.scatter(X[:,1], y, s = 20, c = 'blue', alpha = .5)
    plt.title('DataSet')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()
    
#if __name__ == "__main__":
#    data, label = load_data('./dataset1.txt') 
#    coefs = linear_regression(data, label)       
#    plot_regression(data, label, coefs)
    





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