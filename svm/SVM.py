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
        
        self._W = None
        self._b = None
        
        self._errors = None
        self._alpha = None
    
    
    def _init_params(self, X, y):
        """Intialize all parameters
        """
        n_samples, n_features = X.shape
        
        self._W = np.zeros((n_samples))
        
        self._b = 0.0
        
        self._alpha = np.zeros((n_samples))
        
        self._errors = np.zeros((n_samples))
        
        for i in range(len(n_samples)):
            self._errors[i] = np.dot(self._W, X[:, i]) + self._b - y[i]
    
    
    def _check_kkt(self, W, b, x_i, y_i, alpha_i):
        """make sure if satisfy KKT condition
        """
        
        if alpha_i < 1e-7:
            return y_i * (np.dot(W, x_i) + b) >= 1
        else:
            return abs(y_i * ((np.dot(W, x_i) + b) - 1)) < 1e-7
    
    
    
    def _select_j(self, best_i):
        """
        """
        j_list = [i for i in range(len(self._alpha)) if self._alpha[i] > 0 and i != best_i]
        best_j = -1
        
        # firstly, prior choice j to make sure that 
        # error_i - error_j is the largest
        if len(j_list) > 0:
            max_error = 0
            for j in j_list:
                cur_error = abs(self._error[j] - self._error[best_i])
                if cur_error > max_error:
                    best_j = j
                    max_error = cur_error
        
        else:
            # randomly choose j
            j_list_ = list(range(len(self._alpha)))
            j_list_exclu_best_i = j_list_[:best_i] + j_list_[(best_i + 1):]
            best_j = np.random.choice(j_list_exclu_best_i)
        
        return best_j
            
            
        
                    
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    