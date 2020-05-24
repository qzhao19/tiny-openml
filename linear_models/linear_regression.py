from __future__ import division, print_function, absolute_import

import numpy as np
from scipy import linalg
# from utils import load_data



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
        
        alpha: float, 
                learning rate when update weights
        
        epochs: int,
            max iterations to update weight
    
    """
    def __init__(self, normalize=True, solver='SGD', alpha=0.1, epochs=500, batch_size=1):
        self._normalize = normalize
        self._solver = solver
        self._alpha = alpha
        self._epochs  = epochs 
        self._batch_size = batch_size
        
    
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

    
    def _fit_closed_form(self, X, y):
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
        X_product = np.matmul(X_T, X)
        
        try:
            X_inv = np.linalg.inv(X_product)
        except Exception as error:
            raise error
        
        theta = np.matmul(np.matmul(X_inv, X_T), y)

        return theta
    
    
    def _fit_sgd(self, X, y):
        """Stocastic gradient descend method to fit model 
        
    
        Parameters
        ----------
            X: ndarray-like data of shape [n_samples, n_features], 
                input training data
            y: ndarray_like data of shape [n_samples,]
                Target label
            

        Returns
        -------
        self.class.

        """
        
        # concatenate X and y as a matrix 
        X_y = np.c_[X, y]
        
        # print(X_y.shape)
        n_samples, n_features = X.shape
        
        # init theta
        # theta = np.random.random((n_features, 1))
        theta = np.zeros((n_features, 1), dtype=float)
        
        for _ in range(self._epochs):
            np.random.shuffle(X_y)
            n_iters = int(X_y.shape[0] // self._batch_size)
            for idx in range(n_iters):
                X_y_batch = X_y[self._batch_size * idx : self._batch_size * (idx + 1)]
                X_batch = X_y_batch[:, :-1]
                y_batch = X_y_batch[:, -1:]
                
                # theta - (alpha / n_samples) * (np.dot(np.transpose(X), h - y))
                h = np.dot(X_batch, theta)
                d_w = -2 * np.dot(X_batch.T, h - y_batch) / self._batch_size
                
                theta = theta - (self._alpha / n_samples) * d_w
        
        return theta
        
        
            
        

    def _fit_lstsq(self, X, y):
        """Fit linear model using scipy.linalg.lstsq"""
        
        theta, residues, rank, singular = linalg.lstsq(X, y)
        
        return theta


    def fit(self, X, y):
        """fit linear model"""
        
        n_samples, n_features = X.shape
        
        if self._normalize:
            X = self._normalize_data(X)
            
        X = np.hstack((np.ones((n_samples, 1), dtype=float), X))

        if self._solver == 'CLOSED_FORM':
            self.theta = self._fit_closed_form(X, y)
            
        elif self._solver == 'SGD':
            y = y.reshape(-1,1)
            self.theta = self._fit_sgd(X, y)
            
        elif self._solver == 'LSTSQ':
            self.theta = self._fit_lstsq(X, y)
            
        else:
            raise NotImplementedError

        return self
    
    
    
    def predict(self, X):
    
        if self._normalize:
            X = self._normalize_data(X)
            
        n_samples, n_features = X.shape
    
        return np.dot(X, self.theta)
    
    
    def score(self, y_true, y_pred):
        """
        Return the coefficient of determination R^2 of the prediction. 
        The coefficient R^2 is defined as (1 - u/v), where u is the 
        residual sum of squares ((y_true - y_pred) ** 2).sum() and 
        v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). 
        The best possible score is 1.0 and it can be negative (because 
        the model can be arbitrarily worse).

        Parameters
        ----------
            y_true : float
                true value
            y_pred : float
                prediected value from model
            sample_weight : float
                sample weight of training data.

        Returns
        -------
            R^2 of self.predict(X) wrt. y..
        """
        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        
        return 1- u/v      
    
    

    

