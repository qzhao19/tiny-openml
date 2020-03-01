import numpy as np

def load_data(path):
    """load dataset
    Args:
        path: txt file path
    """
    data, label = [], []
    with open(path, 'r') as file:
        num_features = len(file.readline().split('\t')) - 1
        lines = [line.strip().split('\t') for line in file.readlines()]
    data, label = [], []
    for i in range(len(lines)):
        data.append(lines[i][0:num_features])
        label.append(lines[i][-1])
    
    return np.array(data, dtype=np.float), np.array(label, dtype=np.float)


class LinearRegression(object):
    """LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Args:
        normalize : bool, optional, default False. If True, the regressors X will be normalized 
        before regression by subtracting the mean and dividing by the l2-norm

    """
    def __init__(self, normalize=True):
        self.normalize = normalize

    def _feature_normalize(self, X):
        """normalize data by substracting the mean and dividing sigma
        """
        if not isinstance(X, np.ndarray):
            raise ValueError('Input data is not ndarray type')
        
        n_samples, n_features = X.shape
        x_norm = np.zeros((n_samples, n_features), dtype=X.dtype)

        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)

        for i in range(n_features):
            x_norm[:, i] = (X[:, i] - mu[i])/sigma[i]

        return x_norm
    
    def _compute_cost(self, X, y, theta):
        """compute cost function, e.g. f(x) = (1/N)*{(wx-y)'(wx_y)}
        """
        n_samples = len(y)
        return (np.transpose(X*theta-y))*(X*theta-y)/(2*n_samples)
    
    def _compute_gradient_descent(X, y, theta, alpha, max_steps=100):
        """Compute gradient descent to optimize vals of function
        Args:
            X: ndarray-like data, input data
            y: ndarray_like, label
            theta: model weights
            alpha: float, learning rate when update weights
            max_steps: int, max iterations to update weight
        
        Returns:
            theta and cost of object function J_history

        """
        n_samples, n_features = X.shape


    def test(self, X):
        return self._feature_normalize(X)





 
def linear_regression(X, y):
    """standard linear regression function
    Args:
        X: float, dataset
        y: float, label
    """
    # make sur that matrix shoule be full rank, which means X.T * X inversible
    if np.linalg.det(np.dot(np.transpose(X), X)) == 0:
        raise ValueError('This matrix is singular, cannot do inverse')
    w = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), (np.dot(np.transpose(X), y)))
    return w

def plot_dataset(X, y):
    """plot Scatter figure
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:, 1], y, s=20, c='blue', alpha=0.5)
    plt.title('DataSet')
    plt.xlabel("X")
    plt.xlabel("y")
    plt.show()
    
def plot_regression(X, y, ws):
    """plot linear regression equation
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X[:,1], np.dot(X, ws), c = 'red') 
    ax.scatter(X[:,1], y, s = 20, c = 'blue', alpha = .5)
    plt.title('DataSet')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()
    
if __name__ == "__main__":
    data, label = load_data('./dataset1.txt') 
    coefs = linear_regression(data, label)       
    plot_regression(data, label, coefs)
