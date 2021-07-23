#include "logistic_regression.hpp"
using namespace regression;


template<typename OptimizerType, typename... CallbackTypes>
const arma::vec LogisticRegression::fit(const arma::mat& X, 
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

    LogisticRegressionFunction lrf(X_, y_, lambda);

    optimizer.Optimize(lrf, theta_, callbacks...);
    
    theta_.print("theta: ");
    // this -> theta = theta_;

    return theta_;
}

void LogisticRegression::fit(const arma::mat &X, const arma::vec &y) {
    // fit model
    if (solver == "sgd") {
        ens::SGD<> sgd_optimizer;
        sgd_optimizer.MaxIterations() = max_iter;
        sgd_optimizer.Tolerance() = tol ;
        sgd_optimizer.StepSize() = alpha;
        sgd_optimizer.BatchSize() = batch_size;
        theta = fit(X, y, sgd_optimizer);

    }
    else if (solver == "lbfgs") {
        ens::L_BFGS lbfg_optimizer;
        lbfg_optimizer.NumBasis() = n_basis;
        lbfg_optimizer.MaxIterations() = max_iter;
        lbfg_optimizer.MinGradientNorm() = tol;
        theta = fit(X, y, lbfg_optimizer);
    }
}

const arma::vec LogisticRegression::predict(const arma::mat& X) const{
    // Predict class labels for samples in X.
    // calculate the desicion boundary func
    arma::vec decision_boundary = X * 
        theta.tail_rows(theta.n_elem - 1) + theta(0);

    arma::vec y_pred = 1.0 / (1.0 + arma::exp(decision_boundary));

    for(auto& value:y_pred) {
        if (value > 0.5) {
            value = 1;
        }
        else {
            value = 0;
        }
    }

    return y_pred;
}

const arma::mat LogisticRegression::predict_prob(const arma::mat &X) const {
    // predict prob of class
    arma::vec decision_boundary = X * 
        theta.tail_rows(theta.n_elem - 1) + theta(0);

    arma::vec y_pred = 1.0 / (1.0 + arma::exp(decision_boundary));

    // define a matrix of shape [n_sampels, n_classes], here n_classe 
    // is 2, which means positve case or negative case 
    arma::mat prob(y_pred.n_rows, 2, arma::fill::zeros);

    prob.col(0) = y_pred;
    prob.col(1) = 1 - y_pred;

    return prob;
}

const double LogisticRegression::score(const arma::vec &y_true, 
    const arma::vec &y_pred) const {
    double acc = 0.0;
    std::size_t n_samples = y_true.n_rows;

    for (int i = 0; i < n_samples; i++) {
        bool matched = (y_true(i) == y_pred(i));
        if (matched) {
            acc++;
        }
    }

    return acc / n_samples;
}
