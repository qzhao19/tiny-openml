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
