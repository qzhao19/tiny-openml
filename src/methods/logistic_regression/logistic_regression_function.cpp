#include "logistic_regression_function.hpp"

using namespace regression;


LogisticRegressionFunction::LogisticRegressionFunction(
    const arma::mat &X_, 
    const arma::vec &y_, 
    const double lambda_): X(X_), y(y_), lambda(lambda_) {

    if (X.n_cols != y.n_cols) {
        std::cerr << "The training dataset has " << X.n_cols << " features, but "
        << "the parameters vector got " << y.n_cols << " features." <<std::endl;
    }
    
};



double LogisticRegressionFunction::Evaluate(const arma::mat &theta) const {
    /**
     * The objective function is likelihood function, theta is the vector 
     * of parameters, y is the result, X is the matrix of data
     * 
     * f(w) = sum(y_i * log(sigmoid(w.t() * x_i)) + (1 - y_i) * log(1 - sigmoid(w.t() * x_i)))
     * need to minimize this function.
     * 
     * L2-regularization term is the lambda multiply by the squared l2-norm then divide
     * by 2
    */

    const double regularization;
}