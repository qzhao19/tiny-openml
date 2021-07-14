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