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
    ax.scatter(X[:, 1], y, s=20, c='blue', alpha=0.5) # 描绘样本
    plt.title('DataSet')
    plt.xlabel("X")
    plt.show()
    

plot_dataset(data, label)    
linear_regression(data, label)       

data, label = load_data('./linear_regression/dataset.txt')  
