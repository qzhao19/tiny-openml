import numpy as np
# # from sklearn.datasets import load_iris
















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
