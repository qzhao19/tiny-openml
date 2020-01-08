import numpy as np


def compute_class_mean(data, label, n_classes):
    """Compute class means.

    Args
    ----------
        data : array-like, shape (n_samples, n_features) Input data.
        label : array-like, shape (n_samples,) or (n_samples, n_targets) Target values.
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

def compute_between_class_Sb(data, label, n_classes):
    """Compute between class covariance matrix

    Args
    ----------
        data : array-like, shape (n_samples, n_features) Input data.
        label : array-like, shape (n_samples,) or (n_samples, n_targets) Target values.
        n_classes : int, number of class 
        
    Returns
    -------
        s_b : array-like, shape (n_features, n_features) between class covariance matrix.

    """
    n_samples, n_features = np.shape(data)
    total_mean = np.mean(data, axis=0)
    s_b = np.zeros((n_features, n_features), dtype=float)

    class_means = compute_class_mean(data, label, n_classes)

    for n_class_mean, class_mean in enumerate(class_means):
        n_class = data[label == n_class_mean].shape[0]
        class_mean = np.reshape(class_mean, [n_features, 1])
        total_mean = np.reshape(total_mean, [n_features, 1])

        s_b += n_class * np.dot((class_mean - total_mean), np.transpose(class_mean - total_mean))

    return s_b

