#ifndef METHOD_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
#define METHOD_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
#include "../../prereqs.hpp"

namespace regression {

class LinearRegression {
public:

    /**
     * empty constructor, we initialize the default value of 
     * the lambda and intercedpt 0.0 and true
    */
    LinearRegression(): lambda(0.0), intercept(true) {};

    /**
     * Non-empty constructor, create the model with lamnda and intercept
     * 
     * @param lambda, regularization constant for ridge regression
     * @param intercept, whether or not to include an intercept term
    */
    LinearRegression(const double lambda_, 
                     const bool intercept_): 
        lambda(lambda_), intercept(intercept_) {};

    /**deconstructor*/
    ~LinearRegression() {};

    /**
     * train the lenaer regression model on given dataset, this function 
     * requires to input the dataset ans the relative label
     * 
     * @param X, the matrix of dataset
     * @param y, the label vector
     * 
    */
    void fit(const arma::mat &X, 
             const arma::vec &y);

    /**
     * Train the linear regression model, using the sample weights
     * 
     * @param X, the matrix of dataset 
     * @param y, the label vector
     * @param weights, Individual weights for each sample
    */
    void fit(const arma::mat &X, 
             const arma::vec &y, 
             const arma::vec &weights);

    /**
     * Calculate the predicted value y_pred for test dataset
     * 
     * @param X the test sample
     * @param y the targetedt value 
     * 
     * @return Returns predicted values, ndarray of shape (n_samples,)
    */
    const arma::vec predict(const arma::mat &X) const;

    /**
     * Return the coefficient of determination R^2 of the prediction
     * The coefficeint R^2 is defined as (1 - u/v), where u is the residual 
     * sum of square ((y_pred - y_true) ** 2).sum() and v is the total sum
     * of square ((y_true - y_true.mean()) ** 2).sum() The best possible score 
     * is 1.0 and it can be negative (because the model can be arbitrarily worse). 
     * A constant model that always predicts the expected value of y, disregarding 
     * the input features, would get a R^2 score of 0.0.
    */
    const double score(const arma::vec &y_true, 
                       const arma::vec &y_pred) const;
	
	/**
     * Return the training params theta, 
    */
    const arma::vec& get_theta() const;
    
protected:

    /**
     * Train the linear regression model on the given dataset, and weights.
     * 
     * @param X the matrix of dataset to train model
     * @param y the label to the dataset
     * @param weights observation weights 
    */
    void train(const arma::mat &X, const arma::vec &y, const arma::vec &weights);

private:
    /**
     * @param theta: ndarray_like data of shape [n_samples,]. the parameters that we want ot 
     * calculate, initialized and filled by constructor for the least square method
     * 
     * @param lambda: double, default = 0.0. the Tikhonov regularization parameter for ridge 
     * regression, 0 for linear regression. Regularization improves the conditioning of the 
     * problem and reduces the variance of the estimate.
     * 
     * @param intercept: bool, default = True. whether to fit the intercept for the model. 
     * 
    */
    arma::vec theta;

    double lambda;
    
    bool intercept;

};
}

#endif /*LINEAR_REGRESSION_LINEAR_REGRESSION_HPP*/
