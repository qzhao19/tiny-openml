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


def log_multivar_t_pdf(x, mu, sigma, nu):
    """Evaluate the density function of a multivariate student t
    distribution at the points X
    
    Parameters
    ----------
        x : ndarray-like of shape [n_samples, n_features]
            The point at which to evaluate density.
        mu : ndarray-like of shape [,n_features].T
            The means.
        sigma : ndarray-like of shape [n_features, n_features]
            The covariance  matrix.
        nu : int
            degrees of freedom parameter.

    Returns
    -------
        None.

    """
    n_samples, n_features = np.shape(x)
    
    
    






