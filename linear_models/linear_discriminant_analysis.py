import numpy as np


def compute_class_mean(data, label, n_classes):
    """Compute class means.

    Args
    ----------
        data : array-like, shape (n_samples, n_features)
            Input data.
        label : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        
        n_classes : int, number of class 
        
    Returns
    -------
        means : array-like, shape (n_classes, n_features) Class means.
    """
    class_mean_vector = []
    for c in range(n_classes):
        class_mean_vector.append(np.mean(data[label==c], axis=0))

    return class_mean_vector


def compute_within_class_Sw(data, label, n_classes):
    """Compute within class covariance matrix

    Args
    ----------
        data : array-like, shape (n_samples, n_features)
            Input data.
        label : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        
        n_classes : int, number of class 
        
    Returns
    -------
        s_w : array-like, shape (n_features, n_features) Class covariance matrix.

    """
    n_samples, n_features = np.shape(data)
    s_w = np.zeros((n_features, n_features), dtype=float)
    class_mean_vectors = compute_class_mean(data, label, n_classes)

    for c, class_mean in zip(range(n_classes), class_mean_vectors):
        # define class covariance matrix.
        class_cov = np.zeros((n_features, n_features), dtype=float)
        for x in data[label==c]:
            # print(x.shape)
            x, class_mean = np.reshape(x, [4, 1]), np.reshape(class_mean, [4, 1])
            class_cov += np.dot((x-class_mean), np.transpose(x-class_mean))
        s_w += class_cov
    return s_w


