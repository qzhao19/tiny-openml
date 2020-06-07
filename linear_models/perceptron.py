# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 23:24:09 2020

@author: qizhao
"""


import numpy as np



class Perceptron(object):
    """Perceptron model
    
    Parameters:
        
        max_iters: int, default=100
            The maximum number of passes over the training data (aka epochs). 
        
        
        alpha: float, default=0.0001
            Constant that multiplies the regularization term. 
            
        
    
    
    """
    def __init__(self, max_iters=100, alpha=1e-5):
        self._max_iters = max_iters
        self._alpha = alpha
        self.W = None
    
    def _fit(self, X, y):
        """fit model
        
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
            
            
            
            
            
            
            
            
            
            




