# tiny-openml C++ Library

Welcome to the **Machine Learning C++ Library**! This library provides a comprehensive set of tools and algorithms for machine learning tasks, implemented in C++ for high performance and flexibility. Whether you're working on classification, regression, clustering, or dimensionality reduction, this library has you covered.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [File Structure](#file-structure)
4. [Getting Started](#getting-started)
5. [Examples](#examples)

---

## **Overview**
This library is designed to be modular, efficient, and easy to use. It includes a wide range of machine learning algorithms, from traditional methods like linear regression and k-means clustering to advanced techniques like L-BFGS optimization and Gaussian Mixture Models. The library is organized into a clear and logical file structure, making it easy to extend and customize.

---

## **Features**
- **Core Functionality**:
  - Data loading and splitting.
  - Loss functions (e.g., hinge loss, log loss, mean squared error).
  - Mathematical utilities (e.g., linear algebra, probability, random number generation).
  - Metrics for model evaluation.
  - Preprocessing tools (e.g., transaction encoding).

- **Optimization Algorithms**:
  - Stochastic Gradient Descent (SGD) with various update policies (e.g., momentum, Nesterov momentum).
  - Limited-memory BFGS (L-BFGS) with line search policies (e.g., backtracking, bracketing).
  - Decay policies for learning rate scheduling (e.g., exponential decay, step decay).

- **Machine Learning Methods**:
  - **Clustering**: K-means, Spectral Clustering.
  - **Decomposition**: PCA, Truncated SVD.
  - **Linear Models**: Linear Regression, Logistic Regression, Lasso Regression, Ridge Regression, Perceptron, Softmax Regression.
  - **Mixture Models**: Gaussian Mixture Models.
  - **Naive Bayes**: Gaussian Naive Bayes, Multinomial Naive Bayes.
  - **Neighbors**: K-Nearest Neighbors (KNN).
  - **Rule Models**: Apriori Algorithm.
  - **Tree Models**: Decision Tree Classifier, Decision Tree Regressor.

- **Data Structures**:
  - KD-Tree for efficient nearest neighbor search.
  - Hash Tree for frequent itemset mining.

---

## **File Structure**
The library is organized into the following directories:

- **`core/`**: Core functionality, including data handling, loss functions, mathematical utilities, and optimization algorithms.
  - **`data/`**: Data loading and splitting utilities.
  - **`loss/`**: Loss functions for various tasks.
  - **`math/`**: Mathematical utilities (e.g., linear algebra, probability, random number generation).
  - **`optimizer/`**: Optimization algorithms (e.g., SGD, L-BFGS).
  - **`preprocessing/`**: Data preprocessing tools.
  - **`tree/`**: Tree-based data structures (e.g., KD-Tree, Hash Tree).

- **`methods/`**: Machine learning algorithms organized by task.
  - **`cluster/`**: Clustering algorithms (e.g., K-means, Spectral Clustering).
  - **`decomposition/`**: Dimensionality reduction techniques (e.g., PCA, Truncated SVD).
  - **`linear_model/`**: Linear models (e.g., Linear Regression, Logistic Regression).
  - **`mixture/`**: Mixture models (e.g., Gaussian Mixture Models).
  - **`naive_bayes/`**: Naive Bayes classifiers (e.g., Gaussian Naive Bayes, Multinomial Naive Bayes).
  - **`neighbors/`**: Nearest neighbor algorithms (e.g., KNN).
  - **`rule_model/`**: Rule-based models (e.g., Apriori Algorithm).
  - **`tree/`**: Tree-based models (e.g., Decision Tree Classifier, Decision Tree Regressor).

- **`prereqs.hpp`**: Common prerequisites and dependencies.

---

## **Getting Started**
### **Prerequisites**
 - Eigen3
 - Cmake
