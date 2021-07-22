#include "logistic_regression.hpp"
using namespace regression;


template<typename OptimizerType, typename... CallbackTypes>
void LogisticRegression::fit(const arma::mat& X, 
                             const arma::vec& y, 
                             OptimizerType& optimizer,
                             CallbackTypes&&... callbacks) const {
    
    std::size_t n_samples = X.n_rows;
    std::size_t n_features = X.n_cols;

    arma::mat X_ = X;
    arma::vec y_ = y;
    // logistic regression must have intercept term, because we want to
    // find a decision boundary that is able to seperate 2 class data,
    // intercept term does not exist, the decision boundary no doubt 
    // pass through the origin point.
    
    // init local theat vector
    arma::vec theta_(n_features + 1, arma::fill::zeros);
    X_.insert_cols(n_features, arma::ones<arma::mat>(n_samples, 1));

    // first checkout whether number of elem for vector theta is 
    // equal with number of cols for dataset plus 1
    if (theta_.n_elem != X_.n_cols) {
        X_.insert_cols(n_features, arma::ones<arma::mat>(n_samples, 1));
    }
    
    LogisticRegressionFunction lrf(X, y, lambda);

    optimizer.Optimize(lrf, theta_, callbacks...);

    this -> theta = theta_;

}


void LogisticRegression::fit(const arma::mat &X, const arma::vec &y) {
    // fit model
    if (solver == "sgd") {
        ens::SGD<> sgd_optimizer;
        sgd_optimizer.MaxIterations() = max_iter;
        sgd_optimizer.Tolerance() = tol ;
        sgd_optimizer.StepSize() = alpha;
        sgd_optimizer.BatchSize() = batch_size;

        fit(X, y, sgd_optimizer);

    }
    else if (solver == "lbfgs") {
        ens::L_BFGS lbfg_optimizer;
        lbfg_optimizer.NumBasis() = n_basis;
        lbfg_optimizer.MaxIterations() = max_iter;
        lbfg_optimizer.MinGradientNorm() = tol;
        fit(X, y, lbfg_optimizer);
    }

}



