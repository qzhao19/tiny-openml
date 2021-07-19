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

    // get the number of features ==> nb of cols
    const int n_features = X.n_cols;

    // the regularization term P(w) = lambda / m * norm2(w)
    const double penality = (lambda / n_features) * arma::dot(theta, theta);

    // define the sogmoid function h(x) = 1 / (1 + exp(w'x))
    const arma::vec h = 1.0 / (1.0 + arma::exp(-1 * X * theta));

    // the objective function we want to minimize, so we make positive function as negative
    // -1 / n_features 
    const double cost_fn = (-1 / n_features) * (arma::dot(y, arma::log(h)) + 
                                                arma::dot(1 - y, arma::log(1 - h)));

    return cost_fn + penality;
}


void LogisticRegressionFunction::Gradient(const arma::mat &theta, 
                                          arma::mat &grad) const {
    
    /**
     * Evaluate the gradient of the logistic regression objective function.
     * 
     * @param theta Vector of logistic regression parameters.
     * @param grad Vector to output gradient into.
    */

    


}