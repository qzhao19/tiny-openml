import numpy as np
import operator
import matplotlib.pyplot as plt

def load_data(file_path):
    """Download dating data to testing alogrithm

    Args:
        file_name: input file path which allows to read a txt file
    Returns:
        retuen_mat: a matrix of dating data contains 3 attributs: 
                        1. Number of frequent flyer miles earned per year
                        2. Percentage of time spent playing video games
                        3. Liters of ice cream consumed per week
        label_vect: a vectro conatins labels 
    """
    data = []
    label = []
    with open(file_path, 'r') as file:
        contents = [line.strip().split('\t') for line in file.readlines()]

    for i in range(len(contents)):
        data.append(contents[i][:-1])
        label.append(contents[i][-1])

    return np.array(data, dtype=float), np.array(label, dtype=float).astype(int)



class LogisticRegressionClassifier(object):
    """Logistic Regression classifier.

    """
    def __init__(self):
        # self._alpha = None


    def _sigmoid(self, x):
        """Compute sigmoid function. Here, we must be careful of overflow in exp
        Args:
            x: float, input value
        """
        return np.exp(np.fmin(x, 0)) / (1 + np.exp(-np.abs(x)))

    def _compute_gradient_descent(self, data, label, alpha, max_steps):
        """Compute gradient ascent to optimize max vals of function

        Args:
            data: ndarray-like data, input data
            label: ndarray_like, label
            alpha: float, learning rate when update weights
            max_steps: int, max iterations to update weights

        """
        if (not isinstance(data, np.ndarray)) or (not isinstance(label, np.ndarray)):
            raise ValueError('Data or label shoule be array type')
        
        # get the number of sample to define ths shape of weights and biases
        sample_nums, feature_nums = data.shape
        # init weights as 1
        weights = np.ones((feature_nums, 1), dtype=data.dtype)

        label = label.reshape((sample_nums, 1))
        for step in range(max_steps):
            # forcast predict vals from updated weighted data
            sigmoid_vals = sigmoid(np.dot(data, weights))
            error = label - sigmoid_vals
            # update weights, if is '-', it will compute gradient descent 
            weights = weights + alpha * np.transpose(data).dot(error)
        return weights

    def fit(self, train_X, train_y, alpha=1e-3, max_steps=500):
        return self._compute_gradient_descent(train_X, train_y, alpha, max_steps)

    def predict(self, test_X, test_y, weight):
        """
        """
        



