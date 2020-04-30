# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:19:44 2020

@author: qizhao
"""
import numpy as np
import scipy as sp
from scipy import stats
from scipy.special import gammaln
from sklearn.base import BaseEstimator, ClassifierMixin


def log_multivar_t_pdf(X, mu, Sigma, nu, min_sigma=1e-7):
    """Evaluate the density function of a multivariate student t
    distribution at the points X
    
    Parameters
    ----------
        x : ndarray-like of shape [n, p]
            The point at which to evaluate density.
        mu : ndarray-like of shape [,p].T
            The means.
        sigma : ndarray-like of shape [p, p]
            The covariance  matrix.
        nu : int
            degrees of freedom parameter.

    Returns
    -------
        None.


    """
    n, p = np.shape(X)
    
    # numerator = gammaln((p + nu) / 2)
    # denominator = gammaln(nu / 2) * np.power((nu * np.pi), 1/2) * np.power(np.linalg.det(sigma, 1/2)) * \
    #     np.power(1 + (1 / nu) * np.dot(np.dot((x - mu), np.linalg.inv(sigma)), (x - mu)), (nu + p)/2)
    
    # ret = numerator / denominator
    
    log_ret = 0
    
    try:
        covar_chol = sp.linalg.cholesky(Sigma * nu, lower=True)
    except sp.linalg.LinAlgError:
        
        try:
            covar_chol = sp.linalg.cholesky(Sigma * nu + min_sigma * np.eye(p), 
                                            lower=True)
        except sp.linalg.LinAlgError:
            raise ValueError('"covariances" must be symmetric')
    
    covar_log_det = 2 * np.sum(np.log(np.diag(covar_chol)))
    
    covar_solve = sp.linalg.solve_triangular(covar_chol, (X - mu[None, :]).T, lower=True)
    
    norm = (gammaln((nu + p) / 2.) - gammaln(nu / 2.) - 0.5 * p * np.log(nu * np.pi))
    
    inner = - (nu + p) * 0.5 * np.log1p(np.linalg.norm(covar_solve, ord=2, axis=0)**2)
    
    log_ret = - norm - inner - covar_log_det
    
    return log_ret
    
    
def mvar_t_pdf(X, mu, Sigma, nu):
    """
    

    Parameters
    ----------
        x : ndarray-like of shape [n, p]
            The point at which to evaluate density.
        mu : ndarray-like of shape [,p].T
            The means.
        sigma : ndarray-like of shape [p, p]
            The covariance  matrix.
        nu : int
            degrees of freedom parameter.

    Returns
    -------
        None.

    """
    
    return np.exp(log_multivar_t_pdf(X, mu, Sigma, nu))
    
    

    
    
    
class DiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    """
    """
    def __init__(self, fit_method='MLE', diag_cov=False):
        """
        

        Parameters
        ----------
            fit_method : TYPE, optional
                DESCRIPTION. The default is 'likehood'.
            diag_cov : TYPE, optional
                DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self._fitted =False
        self.fit_method = fit_method
        self.diag_cov = diag_cov
        
    def fit(self, X, y, method=None, alpha=1.0, nu=2, k=1e-3):
        """
            
        Increasing alpha, nu, or k will increase the amount of regularization,
        or in other words, add more weight to the prior belief.  We use the
        data dependent priors mean(X) and diag(cov(X)) for each prior.
        

        Parameters
        ----------
            X : ndarray_like of shape [n_samples, n_features]
                The data matrix.
                
            y : ndarray_like of shape [n_samples]
                Class label for X.
                
            method : string, optional
                The method to fit the model, The default is None.
                    - 'MLE': Fit via maximum likelihood
                    - 'MAP': The Bayesian MAP estimate with conjugate priors
                    - 'Mean': Use Bayesian posterior mean point estimates
                    - 'Bayes': Fit a fully Bayesian model with conjugate priors
                    
            alpha : float, optional
                Prior value for Dir(alpha), ignored for MLE. 
                The default is 1.0.
                
            nu : float, optional
                Covariance pseudo will be p + nu, ignored for ML. 
                The default is 2.
                
            k : float, optional
                 Mean pseudo data, ignored for ML.. The default is 1e-3.

        Returns
        -------
        None.

        """
        
        n_samples, n_features = np.shape(X)
        
        self.n_samples, self.n_features = n_samples, n_features
        
        if len(np.shape(y)) != 1:
            raise ValueError('y must have a single dimension!')
            
        if len(np.shape(X)) != 2:
            raise ValueError('X must have a double dimensions!')
            
        if len(y) != n_samples:
            raise ValueError('X, y length mismatch {} !={}'\
                             .format(str(n_samples), str(len(y))))
        
        classes, n_classes = np.unique(y, return_counts=True)
        
        # get a dict in which display the unique label class and their 
        # frequancies, e.g. {1: 2, 2: 3}
        n_classes = {c: n_classes[i] for i, c in enumerate(classes)}
        
        self.classes, self.n_classes = classes, n_classes
        
        # get the number of unique label
        self.n_categories = len(classes)
        
        
        # Label to class data look up table
        X = {c: np.vstack(X[i, :] for i in range(n_samples) if y[i] == c) 
             for c in classes}
        
        if method is None:
            method = self.fit_method
        
        
        if method == 'MLE':
            self._fit_MLE(X, y)
        
        # bayesian method
        else:   
            self.mu = {c: np.mean(
                np.vstack([np.mean(X[c], axis=0) for c in self.classes]), axis=0) 
                for c in self.classes}
            
            self.sigma = {c: np.diag(
                np.mean(np.vstack([np.std(X[c], axis=0)]), axis=0)) 
                for c in self.classes}
            
            
            self.alpha = alpha
            
            # covar pseudo-data
            self.nu = nu + n_features
            
            # Mean pseudo data
            self.k = k
            
            # if bayesian MAP method
            if method == 'MAP':
                self._fit_MAP(X, y)
            
            
            
            
            
        return self
    
    def predict_prob(self, X):
        """Compute the predicted probability of the data in X for each class
        

        Parameters
        ----------
            X : ndarray_like of shape [n_samples, n_features]
                The data matrix.

        Returns
        -------
            None.

        """
        if not self._fitted:
            raise ValueError('Must fit the model!')
        
        if self._fitted != 
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    def _fit_MLE(self, X, y=None):
        """fit model via maximum likehood method
        

        Parameters
        ----------
            X : ndarray of shape [n_samples, n_features]
                The data matrix. X should be a {label: Array} look up table.
            
            y : ndarray of shape [n_samples], optional
                Class label for X.. The default is None.

        Returns
        -------
            None.

        """
        
        # get probablity of class
        self.class_prob = {c: self.n_classes[c] / self.n_samples for c in self.classes}
        
        # get class means
        self.class_mu = {c: np.mean(X[c], axis=0) for c in self.classes}
        
        # get class Sigma
        self.class_Sigma = {c: np.cov(X[c], bias=True, rowvar =True) for c in self.classes}
        
        if self.diag_cov:
            self.class_Sigma = {c: np.diag(np.diag(self.class_Sigma[c])) for c in self.classes}
        
        self._fitted = 'MLE'
        
        return 
    
    
    
    
    def _bayes_update(self, X, y=None):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        # get the mean of class
        x_mu_per_class = {c: np.mean(X[c], axis=0) for c in self.classes}
        
        alph_hat = {c: self.alpha + self.n_classes[c] for c in self.classes}
        
        k_hat = {c: slef.k + self.n_categories[c] for c in self.classes}
        
        
        
        
    def _fit_MAP(self, X, y):
        """
        

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
    
        return 0






