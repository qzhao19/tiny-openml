# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 23:24:09 2020

@author: qizhao
"""

import numbers
import numpy as np



class Perceptron(object):
    """Perceptron model
    
    Parameters:
        
        max_iters: int, default=100
            The maximum number of passes over the training data (aka epochs). 
        
        
        alpha: float, default=0.0001
            learning rate 
            
        
    
    
    """
    def __init__(self, max_iters=100, alpha=1e-5):
        self._max_iters = max_iters
        self._alpha = alpha
        self._W = None
    
    def _fit(self, X, y):
        """fit model using closed form solution
        
         Parameters
        ----------
            X : ndarray of shape [n_samples, n_featiures]
                Training data.
            y : ndarray of shape (n_samples, ), optional
                Target values. The default is None.
        
        
        """
        n_samples, n_features = X.shape
        
        W = np.zeros((n_features + 1), dtype=X.dtype)
        
        n_iter = 0
        
        while n_iter < self._max_iters:
            idx = np.random.randint(0, y.shape[0] - 1)
            X_ = np.hstack([X[idx], 1])
            y_ = 2 * y[idx] - 1
            
            w_x = np.dot(W, X_)
            
            if w_x * y_ <= 0:
                W += self.alpha * y_ * X_
            
            n_iter += 1
            
        return W
    
    def fit(self, X, y):
        """Fit model
        """
        if (not isinstance(X, np.ndarray)) or (not isinstance(y, np.ndarray)):
            raise ValueError('Data or label must be array type')
        
        if not isinstance(self._alpha, numbers.Number) or self._alpha < 0:
            raise ValueError("Learning rate must be positive; got (alpha=%r)" % self._alpha)
            
        
        if y.ndim > 2:
            raise ValueError("Target y has the wrong shape %s" % str(y.shape))
            
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
            
        n_samples_X, n_features = X.shape

        n_samples_y, n_targets = y.shape
        
        if n_samples_X != n_samples_y:
            raise ValueError("Number of samples in X and y does not correspond:" \
                             " %d != %d" % (n_samples_X, n_samples_y))
            
        self._W = self._fit(X, y)
        
        return self
    
    def predict(self, X):
        # for b
        X = np.hstack([X, np.ones(X.shape[0]).reshape((-1, 1))])
        # activation function for perceptron: sign
        rst = np.array([1 if rst else -1 for rst in np.dot(X, self._W) > 0])
        # np.sign(0) == 0
        # rst = np.sign(np.dot(X, self.w))
        return rst
    
    
            
            
            
            




