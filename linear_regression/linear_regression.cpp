
#include "linear_regression.hpp"
using namespace regression;


void LinearRegression::train(const arma::mat &X, 
                             const arma::vec &y, 
                             const arma::vec &weights) {

    /**
     * Calculate the closed form solution
     * 
     * theta = (X.T * X + lambda * I)^(-1) * X.T * y
    */
    const std::size_t n_samples = X.n_rows;
    
    arma::mat X_ = X;
    arma::vec y_ = y;

    if (this -> intercept) {
        X_.insert_cols(X.n_cols, arma::ones<arma::mat>(n_samples, 1));
    }

    
    if (weights.n_elem > 0) {
        X_ = X_ * arma::diagmat(arma::sqrt(weights));
        y_ = arma::sqrt(weights) % y_;
    }
    

    arma::mat identifty_mat = arma::eye<arma::mat>(X_.n_cols, X_.n_cols);
    arma::mat pseudo_inv = (X_.t() * X_ + this -> lambda * identifty_mat);


    this -> theta = arma::inv(pseudo_inv) * X_.t() * y_;

};


void LinearRegression::predict(const arma::mat &X, arma::vec &y) {
    /**
     * y_pred = X * theta
    */

    if (intercept) {
        if (X.n_cols != theta.n_rows - 1) {
            return ;
        }

        // if intercept, need split the intercept term from theta vector
        y = X * theta.subvec(0, theta.n_rows - 2);

        y += theta(theta.n_rows - 1);
    }
    else {
        y = X * theta;
    }
}


void LinearRegression::fit(const arma::mat &X, 
                           const arma::vec &y, 
                           const arma::vec &weights) {

    train(X, y, weights);
}

void LinearRegression::fit(const arma::mat &X, 
                           const arma::vec &y) {

    train(X, y, arma::vec());
}




const arma::vec& LinearRegression::get_theta() const {
    return this -> theta;
}

double LinearRegression::get_lambda() const {
    return this -> lambda;
}

bool LinearRegression::get_intercept() const {
    return this -> intercept;
}

