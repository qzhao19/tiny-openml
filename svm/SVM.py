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
        
        supprot_vectors: ndarray of shape [n_samples]
            The supports vectors
        
        W: ndarray of shape [n_samples]
            model's weights
        
        b: ndarray of shape [n_samples]
            model's biases
        
        errors: ndarray of shape [n_samples]
            The errors, the difference between true values and predicted values
        
        alpha: ndarray of shape [n_samples]
            Lagrange multipliers

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
    
    
    def _satisty_kkt(self, W, b, x_i, y_i, alpha_i):
        """make sure if satisfy KKT condition
        
                1. dL/dw = 0, dL/db = 0
                2. alpha_i * (1 - y_i * (w_T * X_i + b)) = 0
                3. alpha_i >= 0
                4. 1 - y_i * (w_T*X_i + b) <= 0
        """
        
        if alpha_i < 1e-7:
            return y_i * (np.dot(W, x_i) + b) >= 1
        else:
            return abs(y_i * ((np.dot(W, x_i) + b) - 1)) < 1e-7
    
    
    
    def _select_idx_j(self, best_i):
        """Return an index j which we got the best error_i - error_j 
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
    
    
    
    def _fit(self, X, y):
        """Fit model
        """
        # initialize params
        self._init_params(X, y)
        for _ in range(self._max_iters):
            # set a flag to check if all points satisfy kkt condiction
            is_satisfied_kkt = True
            for i in range(len(self._alpha)):
                x_i = X[i, :]
                y_i = y[i]
                old_alpha_i = self._alpha[i]
                old_error_i = self._error[i]
                # choose all points i what break kkt condition
                if not self._satisty_kkt(self._W, self._b, x_i, y_i, old_alpha_i):
                    is_satisfied_kkt = False
                    # select point index i we could get the best for error_i - error_j
                    best_j = _select_idx_j(i)
                    # get x_j, y_j, old_aplha_j and old_eooro_j
                    x_j = X[best_j, :]
                    y_j = y[best_j]
                    old_alpha_j = self._alpha[best_j]
                    old_error_j = self._error[best_j]
                    
                    # update parameters
                    # 1. got optimized alpha_2 that not clipped
                    theta = np.dot((x_i -x_j), (x_i - x_j))
                    # if x_i and x_j are closed, continue
                    if theta < 1e-3:
                        continue
                    opt_alpha_j = old_alpha_j + y_j * (old_error_i - old_error_j) * theta
                    
                    # 2. we do clip alpha to get new_alpha_2
                    if y_i == y_j:
                        if opt_alpha_j < 0:
                            new_alpha_j = 0
                        elif 0 <= opt_alpha_j < old_alpha_i + old_alpha_j:
                            new_alpha_j = opt_alpha_j
                        else:
                            new_alpha_j = old_alpha_i + old_alpha_j
                    else:
                        if opt_alpha_j <  max(0, old_alpha_j - old_alpha_i):
                            new_alpha_j = max(0, old_alpha_j - old_alpha_i)
                        else:
                            new_alpha_j = opt_alpha_j
                    # if abs(new_alph_i - new_alpha_j) < threadhold
                    # we continues
                    if np.abs(new_alpha_j - old_alpha_j) < 1e-5:
                        continue
                    
                    # 3. get new alpha_1
                    new_alpha_i = old_alpha_i + y_i * y_j * (old_alpha_j - new_alpha_j)
                    
                    # 4. update W
                    
        
        def fit(self, X, y):
            
            return self
                    
                    
                    
                
        
        
        
                    
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    