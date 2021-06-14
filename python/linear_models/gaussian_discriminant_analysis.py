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


def log_multivariate_t_pdf(X, mu, Sigma, nu):
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
    
    min_sigma=1e-7
    
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
    
    
def multivariate_t_pdf(X, mu, Sigma, nu):
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
    
    return np.exp(log_multivariate_t_pdf(X, mu, Sigma, nu))
    
    

    
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
            if method == 'BAYES_MAP':
                self._fit_MAP(X, y)
                
            elif method == 'BAYES_POSTERIOR_MEAN':
                self._fit_bayes_mean(X, y)
                
            elif method == 'BAYES_FULL':
                self._fit_bayes_full(X, y)
            
            else:
                raise NotImplementedError
                
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
        
        if self._fitted != 'BAYES_FULL':
            # gaiussian posterior
            def calc_prob(c):
                try:
                    return self.class_prob[c] * \
                        stats.multivariate_normal.pdf(X, 
                                                      mean=self.class_mu[c],
                                                      cov=self.class_Sigma[c])
                
                
                except np.linalg.LinAlgError as error:  # Singular matrix
                    print("LinAlgError on normal.pdf mean = %s, cov = %s"
                          %(self.class_mu[c], self.class_Sigma[c]))
                    raise error
        else:
            # T-distribustion posterior
            def calc_prob(c):
                try:
                    return self.class_prob[c] * \
                        multivariate_t_pdf(X, mu=self.class_mu[c],
                                           Sigma=self.class_Sigma[c], 
                                           nu=self.nu_post[c])
                
                
                except np.linalg.LinAlgError as error:  # Singular matrix
                    print("LinAlgError on normal.pdf mean = %s, cov = %s"
                          %(self.class_mu[c], self.class_Sigma[c]))
                    raise error
        
        
        density = {}
        normalizer = np.zeros(X.shape[0])
        
        for c in self.classes:
            try:
                prob = calc_prob(c)
            except np.linalg.LinAlgError:
                prob = np.nan * np.empty(X.shape[0])
                density[c] = prob
                continue
            
            density[c] = prob
            normalizer += prob
        
        for c in self.classes:
            density[c] = density[c] / normalizer
        
        density = np.vstack(density[c] for c in density.keys()).T
        
        return density
    
    
    
    def predict_log_proba(self, X):
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
        
        if self._fitted != 'BAYES_FULL':
            # gaiussian posterior
            def calc_log_prob(c):
                try:
                    return self.class_prob[c] * \
                        stats.multivariate_normal.logpdf(X, 
                                                         mean=self.class_mu[c],
                                                         cov=self.class_Sigma[c])
                
                
                except np.linalg.LinAlgError as error:  # Singular matrix
                    print("LinAlgError on normal.pdf mean = %s, cov = %s"
                          %(self.class_mu[c], self.class_Sigma[c]))
                    raise error
        else:
            # T-distribustion posterior
            def calc_log_prob(c):
                try:
                    return self.class_prob[c] * \
                        log_multivariate_t_pdf(X, 
                                               mu=self.class_mu[c],
                                               Sigma=self.class_Sigma[c], 
                                               nu=self.nu_post[c])
                
                
                except np.linalg.LinAlgError as error:  # Singular matrix
                    print("LinAlgError on normal.pdf mean = %s, cov = %s"
                          %(self.class_mu[c], self.class_Sigma[c]))
                    raise error
        
        
        log_density = {}
        # normalizer = np.zeros(X.shape[0])
        
        for c in self.classes:
            try:
                log_prob = calc_log_prob(c)
            
            except np.linalg.LinAlgError:
                log_prob = np.nan * np.empty(X.shape[0])
                log_density[c] = np.zeros_like(log_prob)
                continue
            
            log_density[c] = log_prob
        
      
        
        log_density = np.vstack(log_density[c] for c in log_density.keys()).T
        
        return log_density
        
    
    def predict(self, X):
        """Predict the class label of each sample in X by picking the most
        probable label"""
        
        prob = self.predict_prob(X)
        
        preds = self.classes[np.argmax(prob, axis=1)]

        return preds
    
    
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
        """update parameters via bayesian method
        """
        classes = self.classes
        
        # get the oberserved mean of class
        x_bar = {c: np.mean(X[c], axis=0) for c in classes}
        
        # x_hat which meaning oberved values, e.g. x_bar: observed total means
        alpha_hat = {c: self.alpha + self.n_classes[c] for c in classes}
        
        k_hat = {c: self.k + self.n_categories[c] for c in classes}
        
        # mu_hat: total mean, it is diffrent between mu_hat and class_mu 
        mu_hat = {c: (self.k * self.mu[c] + self.n_classes[c] * x_bar[c]) 
                  / k_hat[c] for c in classes}
        
        Sigma_hat = {c: self.Sigma[c] + self.n_classes[c] * np.cov(X[c], 
                                                                   bias=True, 
                                                                   rowvar=False) + 
                     (self.k * self.n_classes[c] / (self.k + self.n_classes[c])) * 
                     np.outer(x_bar[c] - self.mu[c], x_bar[c] - self.mu[c]) 
                     for c in classes}
        nu_hat = {c: self.nu + self.n_classes[c] for c in classes}
        
        return alpha_hat, k_hat, mu_hat, Sigma_hat, nu_hat
    
    
    
    
    
    def _fit_MAP(self, X, y=None):
        """Fits model via Bayesian MAP

        """
        if self.diag_cov:
            raise ValueError("Diagonal covariance restriction is only "
                             "supported for maximum likelihood.")
        
        
        classes = self.classes
        n_features = self.n_features
        
        alpha_hat, k_hat, mu_hat, Sigma_hat, nu_hat = self._bayes_update(X, y)
        
        class_prob_norm = sum(alpha_hat[c] for c in classes) + self.n_categories
        
        self.class_prob = {c: (alpha_hat[c] - 1) / class_prob_norm for c in classes}
        
        self.class_Sigma = {c: Sigma_hat[c] / (nu_hat[c] + n_features + 1) for c in classes}
        
        self.mu = mu_hat
        
        self._fitted = 'BAYES_MAP'
        
        return 

    
    def _fit_bayes_mean(self, X, y=None):
        """Fits model via Bayesian posterior mean.

        """
        if self.diag_cov:
            raise ValueError("Diagonal covariance restriction is only "
                             "supported for maximum likelihood.")
        
        
        classes = self.classes
        n_features = self.n_features
        
        alpha_hat, k_hat, mu_hat, Sigma_hat, nu_hat = self._bayes_update(X, y)
        
        class_prob_norm = sum(alpha_hat[c] for c in classes) + self.n_categories
        
        self.class_prob = {c: (alpha_hat[c] - 1) / class_prob_norm for c in classes}
        
        self.class_Sigma = {c: Sigma_hat[c] / (nu_hat[c] - n_features - 1) for c in classes}
        
        self.mu = mu_hat
        
        self._fitted = 'BAYES_POSTERIOR_MEAN'
        
        return 


    def _fit_bayes_full(self, X, y=None):
        """Fits model via fully Bayesian model with conjugate priors

        """
        if self.diag_cov:
            raise ValueError("Diagonal covariance restriction is only "
                             "supported for maximum likelihood.")
        
        
        classes = self.classes
        n_features = self.n_features
        
        alpha_hat, k_hat, mu_hat, Sigma_hat, nu_hat = self._bayes_update(X, y)
        
        class_prob_norm = sum(alpha_hat[c] for c in classes) + self.n_categories
        
        self.class_prob = {c: (alpha_hat[c] - 1) / class_prob_norm for c in classes}
        
        # self.class_Sigma = {c: Sigma_hat[c] / (nu_hat[c] + n_features + 1) for c in classes}
        
        self.class_Sigma = {c: ((k_hat[c] + 1) / (k_hat[c] * (nu_hat[c] - n_features + 1))) * 
                            Sigma_hat[c] for c in classes}
        
        self.mu = mu_hat
        
        self.nu_post = {c: nu_hat[c] - n_features + 1 for c in classes}
        
        self._fitted = 'BAYES_FULL'
        
        return 

