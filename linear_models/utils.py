# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:22:19 2020

@author: qizhao
"""
import numpy as np
import matplotlib.pyplot as plt

# def load_data(path):
#     """load dataset
#     Args:
#         path: txt file path
#     """
#     data, label = [], []
#     with open(path, 'r') as file:
#         num_features = len(file.readline().split('\t')) - 1
#         lines = [line.strip().split('\t') for line in file.readlines()]
#     data, label = [], []
#     for i in range(len(lines)):
#         data.append(lines[i][0:num_features])
#         label.append(lines[i][-1])
    
#     return np.array(data, dtype=np.float), np.array(label, dtype=np.float)


def load_data(file_path, sep = '\t'):
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
        contents = [line.strip().split(sep) for line in file.readlines()]

    for i in range(len(contents)):
        data.append(contents[i][:-1])
        label.append(contents[i][-1])

    return np.array(data, dtype=float), np.array(label, dtype=float)



# def linear_regression(X, y):
#     """standard linear regression function
#     Args:
#         X: float, dataset
#         y: float, label
#     """
#     # make sur that matrix shoule be full rank, which means X.T * X inversible
#     if np.linalg.det(np.dot(np.transpose(X), X)) == 0:
#         raise ValueError('This matrix is singular, cannot do inverse')
#     w = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), (np.dot(np.transpose(X), y)))
#     return w

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
    
#if __name__ == "__main__":
#    data, label = load_data('./dataset1.txt') 
#    coefs = linear_regression(data, label)       
#    plot_regression(data, label, coefs)