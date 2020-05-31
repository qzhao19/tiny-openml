# -*- coding: utf-8 -*-
"""
Created on Sun May 31 22:07:51 2020

@author: qizhao
"""

import numpy as np


def HardMarginSVM(object):
    """hard margin svm model
    

    Parameters
    ----------
        max_iters: int
            The number of iteration to be run
        
        supprot_vectors: ndarray
            The supports vectors
        
        weights: ndarray of shape [n_samples]
            model's weights
        
        biases: ndarray of shape [n_samples]
            model's biases
        
        errors: ndarray of shape [n_samples]
            The errors, the difference between true values and predicted values
        
        alpha: float
            Regularization parameter, it must be strictly positive.

    Returns
    -------
    None.

    """
    
    
    def __init__(self, max_iters=100):
        self._max_iters = max_iters
        
        self.support_vectors = None
        
        self._weights = None
        self._biases = None
        
        self._errors = None
        self._alpha = None
    
    
    def init_params(self, X, y):
        """Intialize all parameters
        """
        n_samples, n_features = X.shape
        
        self._weights = np.zeros((n_samples))
        self._biases = 0.0
        
        self._alpha = np.zeros((n_samples))
        
        self._errors = np.zeros((n_samples))
        
        for i in range(len(n_samples)):
            self._errors[i] = np.dot(self._weights, X[:, i]) + self._biases - y[i]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    