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


def log_multivar_t_pdf(x, mu, sigma, nu, min_sigma=1e-7):
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
    n, p = np.shape(x)
    
    # numerator = gammaln((p + nu) / 2)
    # denominator = gammaln(nu / 2) * np.power((nu * np.pi), 1/2) * np.power(np.linalg.det(sigma, 1/2)) * \
    #     np.power(1 + (1 / nu) * np.dot(np.dot((x - mu), np.linalg.inv(sigma)), (x - mu)), (nu + p)/2)
    
    # ret = numerator / denominator
    
    log_ret = 0
    
    try:
        covar_chol = sp.linalg.cholesky(sigma, lower=True)
    except linalg.LinAlgError:
        
        try:
            covar_chol = sp.linalg.cholesky(sigma + min_sigma * np.eye(p), 
                                            lower=True)
        except linalg.LinAlgError:
            raise ValueError('"covariances" must be symmetric')
    
    covar_log_det = np.sum(np.log(np.diag(covar_chol)))
    
    
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
                Prior value for Dir(alpha), ignored for MLE. The default is 1.0.
                
            nu : float, optional
                Covariance pseudo will be p + nu, ignored for ML. The default is 2.
                
            k : float, optional
                 Mean pseudo data, ignored for ML.. The default is 1e-3.

        Returns
        -------
        None.

        """
        
        n_samples, n_features = np.shape(X)
        
        self.n, self.p = n_samples, n_features
        
        if len(np.shape(y)) != 1:
            raise ValueError('y must have a single dimension!')
            
        if len(np.shape(X)) != 2:
            raise ValueError('X must have a double dimensions!')
            
        if len(y) != n_samples:
            raise ValueError('X, y length mismatch {} !={}'\
                             .format(str(n_samples), str(len(y))))
        
        classes, n_classes = np.unique(y, return_counts=True)
        
        n_classes = {c: n_classes[i] for i, c in enumerate(classes)}
        
        self.classes, self.n_classes = classes, n_classes
        
        
        
        
        return 0
        
    
    def _fit_likehood(self, X, y=None):
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
        return 0
    
    
    
    
    
    
    
    
    






