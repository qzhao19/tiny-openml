import numbers
import numpy as np
from scipy import linalg
# # from sklearn.datasets import load_iris



def svd_flip(U, V, U_based_decision=True):
    """Sign correction to make sure e deterministic output from SVD. 
    Adjusts the columns of u and the rows of v such that the loadings 
    in the columns in u that are largest in absolute value are always positive.
    
    

    Parameters
    ----------
        U : ndarray 
            u and v are the output of `linalg.svd` or
        :func:`sklearn.utils.extmath.randomized_svd`, 
        with matching inner dimensions so one can compute
        `np.dot(u * s, v)`.
        
        V : TYPE
            u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, 
        with matching inner dimensions so one can compute 
        `np.dot(u * s, v)`.
        
        U_based_decision : boolean, optional,
            If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent. The default is True.

    Returns
    -------
         u_adjusted, v_adjusted : arrays with the same dimensions as the input.
         
    """
    
    if U_based_decision:
        # columns of U, rows of V
        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        V *= signs[:, np.newaxis]
    else:
        # columns of V, rows of U
        max_abs_rows = np.argmax(np.abs(V), axis=1)
        signs = np.sign(V[range(V.shape[1]), max_abs_rows])
        U *= signs
        V *= signs[:, np.newaxis]
        
    return U, V 




class PCA(object):
    """Principal component analysis

    Linear dimensionality reduction using Singular Value Decomposition of the
    data to project it to a lower dimensional space.
    
    Parameters:
        n_components: int
            number of the components keep 
        
        svd_solver: str
            Methods for solving svd 'full', 'arpack', 'randomized'
            
            - 'full': run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
            
            - 'randomized': run randomized SVD by the method of Halko et al.
        
    """

    def __init__(self, n_components=2, solver=None):
        self.n_components = n_components
        self.solver = solver
    
    
    def _fit_eig(self, X, n_components):
        """fit the model by eigenvector decomposition"""
        
        X_mu = np.mean(X, axis=0, keepdims=True)
        X -= X_mu
        
        S = np.dot(X.T, X)
        
        eig_values, eig_vectors = np.linalg.eig(S)
        
        idx = np.abs(eig_values).argsort()[::-1]
        
        # sorted_eig_values = eig_values[idx]
        
        sorted_eig_vectors = eig_vectors[:, idx]
        
        # get the components of X
        components = sorted_eig_vectors[:, :n_components]
        
        # compute projection matrix 
        # projection_mat = components @ np.linalg.inv(components.T, components) @ components.T 
        
        # X_transforme = (projection_mat @ X.T).T
        
        return components
        
    
    
    def _fit_full(self, X, n_components):
        """Fit the model by computing full SVD on X"""
        
        n_samples, n_features = X.shape
        
        if not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError("n_components=%r must be between 0 and "
                             "min(n_samples, n_features)=%r with "
                             "svd_solver='full'"
                             % (n_components, min(n_samples, n_features)))
        
        else:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError('n_components=%r must be of type int')
            
        
        # center data
        X_mu = np.mean(X, axis=0, keepdims=True)
        X -= X_mu
        
        U, S, V = linalg.svd(X, full_matrices=False)
        
        U, V = svd_flip(U, V)
        
        return U, S, V
    
    
        

    def _fit(self, X):
        """fit model"""
        
        return 




# class PCA(object):
#     """Principal component analysis

#     Linear dimensionality reduction using Singular Value Decomposition of the
#     data to project it to a lower dimensional space.

#     """
#     def __init__(self, n_components):
#         self.n_components = n_components

#     def _compute_mean(self, data):
#         """compute mean of each dimension

#         Args:
#             data : array-like, shape (n_samples, n_features) Input data.

#         Returns:
#             array-like, shape (n_samples, n_features)
#         """
#         if len(np.shape(data)) != 2:
#             raise ValueError('Training data shape must be 2!')

#         return np.mean(data, axis=0, keepdims=True)

#     def _compute_cov_matrix(self, data):
#         """Compute covariance matrix of data

#         Args:
#             data : array-like, shape (n_features, n_features) Input data.

#         Returns:
#             array-like, shape (n_features, n_features)
#         """

#         centring_data = self._compute_mean(data)

#         return np.dot(np.transpose(centring_data), centring_data)

#     def fit(self, data):
#         """pca function

#         Args:
#             X : array-like, shape (n_samples, n_features) Input data.
#         """

#         cov_matrix = self._compute_cov_matrix(data)

#         print(cov_matrix.shape)
#         eig_values, eig_vectors = np.linalg.eig(cov_matrix)

#         indexs = np.argsort(-eig_values)[:self.n_components]
#         sorted_eig_values = eig_values[indexs]
#         sorted_eig_vectors = eig_vectors[:, indexs]

#         # print(sorted_eig_values)
#         # print(np.shape(sorted_eig_vectors))
#         data_ndim = np.dot(data, sorted_eig_vectors)
#         return data_ndim
