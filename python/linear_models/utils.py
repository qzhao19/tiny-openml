# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:22:19 2020

@author: qizhao
"""
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path, sep = '\t'):
    """Download dating data to testing alogrithm

    Args:
        file_name: input file path which allows to read a txt file
    Returns:
        retuen_mat: a matrix of data
        label_vect: a vectro conatins labels 
    """
    data = []
    label = []
    with open(file_path, 'r') as file:
        contents = [line.strip().split(sep) for line in file.readlines()]

    for i in range(len(contents)):
        data.append(contents[i][:-1])
        label.append(contents[i][-1])

    return np.array(data, dtype=float), np.array(label, dtype=float).reshape(-1, 1)


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
    