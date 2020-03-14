import numpy as np
from scipy.sparse import linalg as sp_linalg




class Ridge(object):
    """Linear least squares with l2 regularization.
    
    Minimizes the objective function: ||y - Xw||^2_2 + alpha * ||w||^2_2
    This model solves a regression model where the loss function is
    the linear least squares function and regularization is given by
    the l2-norm.
    
    Parameters
    ----------
        C: float, default is 1.0
           Regularization strength; must be a positive float. Regularization
           improves the conditioning of the problem and reduces the variance of
           the estimates. Larger values specify stronger regularization.
           
       normalize: boolean, optional, default False
           If True, the regressors X will be normalized before regression.
           
        solver: string, default value is {auto, svd}
            Solver to use in the computational routines:
            
    
    
    """

    def __init__(self, alpha = 1.0, normalize = False, solver = 'auto'):
        self._alpha = alpha
        self._normalize = normalize
        self._solver = solver
        
    def _normalize(self, X, y):
        """Normalize data 
        
            X = (X - mu)/sigma
            y = y - mu
        

        Parameters
        ----------
            X : ndarray of shape [n_samples, n_features]
                Training vector.
            y : ndarray of shape [n_samples, 1]
                Target label vector.

        Returns
        -------
            Normalized training matrix and target vector

        """
        
        return (X - np.mean(X, axis = 1))/np.std(X, axis=1), y - np.mean(y)


    def _solve_sparse_cg(self, X, y, max_iter=None, tol = 1e-3):
        """

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        max_iter : TYPE, optional
            DESCRIPTION. The default is None.
        tol : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        return 0
    
    def _solve_lsqr(self, X, y, tol = 1e-3, max_iter=None):
        """‘lsqr’ uses the dedicated regularized least-squares routine 
        scipy.sparse.linalg.lsqr. It is the fastest and uses an 
        iterative procedure.
        
        Parameters
        ----------
            X : ndarray-like matrix [n_samples, n_features]
                Representation of an m-by-n matrix.
            y : ndarray-like of shape [n_samples, ]
                Target values.
                
            tol : float, optional
                Stopping tolerances..
                
            max_iter : int, optional
                Explicit limitation on number of iterations. The default is None.
            
        """
        n_samples, n_features = X.shape
        
        # coefs = np.zeros((n_features, 1))
        
        sqrt_alpha = np.sqrt(self._alpha)
        
        coefs = sp_linalg.lsqr(X, y, damp=sqrt_alpha, atol=tol, btol=tol, iter_lim=max_iter)
        
        return coefs
        
        
        
        


# def ridge(A, b, alphas):
#     """Return coefficients for regularized least squares

#          min ||A x - b||^2 + alpha ||x||^2

#     Parameters
#     ----------
#     A : array, shape (n, p)
#     b : array, shape (n,)
#     alphas : array, shape (k,)

#     Returns
#     -------
#     coef: array, shape (p, k)
#     """
#     U, s, Vt = linalg.svd(X, full_matrices=False)
#     d = s / (s[:, np.newaxis].T ** 2 + alphas[:, np.newaxis])
#     return np.dot(d * U.T.dot(y), Vt).T


# class RidgeRegression(object):
#     """Linear least squares with l2 regularization.
    
#     Minimizes the objective function: ||y - Xw||^2_2 + alpha * ||w||^2_2
#     This model solves a regression model where the loss function is
#     the linear least squares function and regularization is given by
#     the l2-norm.
    
#     Parameters
#     ----------
#         C: float, default is 1.0
#            Regularization strength; must be a positive float. Regularization
#            improves the conditioning of the problem and reduces the variance of
#            the estimates. Larger values specify stronger regularization.
           
#        solver: 
    
    
#     """
#     def __init__(self, alpha = 1.0, solver = 'lsqr'):
#         self._alpha = alpha
#         self._solver = solver
    
#     def _cost_fn(self, theta, X, y):
#         """compute cost function, e.g. f(x) = (1/N)*{(wx-y)'(wx_y)}
#         """
#         n_samples = len(y)
#         return (np.transpose(X * theta - y)) * (X * theta - y)/(2 * n_samples) + self._alpha * np.dot(np.transpose(theta), theta)
    
    
    
    
    
    
    
    
    
    
    
    
    
