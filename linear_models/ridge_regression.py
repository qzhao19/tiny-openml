import numpy as np
from scipy import optimize


class RidgeRegression(object):
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
           
       solver: 
    
    
    """
    def __init__(self, C = 1.0, solver = 'lsqr'):
        self._C = C
        self._solver = solver
    
    def _cost_fn(self, theta, X, y)
        
    
