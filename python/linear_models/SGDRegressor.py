# -*- coding: utf-8 -*-
"""
Created on Sun May 31 23:47:41 2020

@author: qizhao
"""

import numbers
import numpy as np



class SGDRegressor(object):
    """Linear model fitted by minimizing a regularized empirical loss with SGD
    
    Parameters:
        
        max_iters: int, default=100
            The maximum number of passes over the training data (aka epochs). 
        
        penalty: ‘l2’, ‘l1’, ‘elasticnet’
            The penalty (aka regularization term) to be used.
        
        alpha: float, default=0.0001
            Constant that multiplies the regularization term. 
            
        l1_ratiofloat, default=0.15
            The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. 
            l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. 
            Only used if penalty is ‘elasticnet’.
    
    """
    def __init__(self,
                 max_iters=100, 
                 penality=None, 
                 alpha=1e-3, 
                 l1_ratio=0.15, 
                 batch_size=3):
        
        self._W = None
        
        self._max_iters = max_iters
        self._penality = penality
        self._alpha = alpha
        
        self._l1_ratio = l1_ratio
        
        self._batch_size = batch_size

    def _fit(self, X, y):
        """Fit the model"""
        
        
        X_y = np.c_[X, y]
        
        n_samples, n_features = X.shape
        
        # X = np.c_[X, np.ones((n_samples, 1), dtype=X.dtype)]
        
        W = np.random.randn(n_features, 1)
        

        for step in range(self._max_iters):
            # shuffle X_y
            np.shuffle(X_y)
            n_iters = int(n_samples / self._batch_size)
            for idx in range(n_iters):
                X_y_batch = X_y[self._batch_size * idx : self._batch_size * (idx + 1)]
                X_batch = X_y_batch[:, :-1]
                y_batch = X_y_batch[:, -1]
            
                h = np.dot(X_batch, W)
                d_W = 2 * np.dot(X_batch.T, h - y_batch) / self._batch_size
                
                learning_rate = 5 / ((step * n_samples + idx) + 50)
                
                if self._penality is not None:
                    # ridge regression
                    if self._penality == 'l2':
                        d_W += self._alpha * W
                    
                    # lasso regression
                    elif self._penality == 'l1':
                        d_W += self._alpha * np.sign(W)
                    
                    elif self._penality == 'elasticnet':
                        d_W += (self._alpha * self._l1_ratio * np.sign(W) + 
                                self._alpha * (1 - self._l1_ratio) * W)
                    
                    W = W * learning_rate - d_W
        
        return W
                            
                    
                
    def fit(self, X, y):
        """
        """
        if (not isinstance(X, np.ndarray)) or (not isinstance(y, np.ndarray)):
            raise ValueError('Data or label must be array type')
        
        if not isinstance(self._alpha, numbers.Number) or self._alpha < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self._alpha)
            
        if not isinstance(self._l1_ratio, numbers.Number):
            raise ValueError('The Elastic Net mixing parameter must be a float')
        else:
            if self._l1_ratio <0 or self._l1_ratio >1:
                raise ValueError('The Elastic Net mixing parameter must be between 0 and 1')
        
        
        if y.ndim > 2:
            raise ValueError("Target y has the wrong shape %s" % str(y.shape))
            
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        
        self._W = self._fit(X, y)
        
        return self
    
    def predict(self, X):
        """predict test dataset"""
        
        if self.W is None:
            raise RuntimeError('cant predict before fit')
        y_pred = X.dot(self.W)
        return y_pred
    
    
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
      
        




