#ifndef LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
#include <iostream>
#include <armadillo>




namespace regression {


class LinearRegression {
public:

    /**empty constructor*/
    LinearRegression(): lambda(0.0), intercept(true) {};

    /**constructor to initialze lamnda and intercept*/
    LinearRegression(const double lambda_, bool intercept_): 
        lambda(lambda_), intercept(intercept_) {};

    void fit(arma::mat &X, arma::rowvec &y);

    void fit(arma::mat &X, arma::rowvec &y, arma::rowvec &weights);

    

protected:

    void train(arma::mat &X, arma::rowvec &y, arma::rowvec &weights);

    const arma::rowvec& predict(arma::mat &X);


private:
    /**
     * @params theta: ndarray_like data of shape [n_samples,]. the parameters that we want ot 
     * calculate, initialized and filled by constructor for the least square method
     * 
     * @params lambda: double, default = 0.0. the Tikhonov regularization parameter for ridge 
     * regression, 0 for linear regression. Regularization improves the conditioning of the 
     * problem and reduces the variance of the estimate.
     * 
     * @params intercept: bool, default = True. whether to fit the intercept for the model. 
     * 
    */
    
    arma::vec theta;
    double lambda;
    bool intercept;




};

}


#endif /*LINEAR_REGRESSION_LINEAR_REGRESSION_HPP*/
