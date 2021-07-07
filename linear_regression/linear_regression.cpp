
#include "linear_regression.hpp"
using namespace regression;


void LinearRegression::train(const arma::mat &X, 
                             const arma::vec &y, 
                             const arma::vec &weights) {


    const std::size_t n_samples = X.n_rows;
    const std::size_t n_dims = X.n_cols;

    arma::mat X_ = X;
    arma::vec y_ = y;

    if (this -> intercept) {
        X_.insert_cols(0, arma::ones<arma::mat>(n_samples, 1));
    }

    if (weights.n_elem > 0) {
        X_ = X_ * arma::diagmat(arma::sqrt(weights));
        y_ = arma::sqrt(weights) % y_;
    }
    

    arma::mat identifty_mat = arma::eye<arma::mat>(n_dims, n_dims);
    arma::mat pseudo_inv = (X_.t() * X_ + this -> lambda * identifty_mat);


    this -> theta = arma::inv(pseudo_inv) * X_.t() * y_;

};



void LinearRegression::fit(const arma::mat &X, 
                           const arma::vec &y, 
                           const arma::vec &weights) {

    train(X, y, weights);
}

void LinearRegression::fit(const arma::mat &X, 
                           const arma::vec &y) {

    train(X, y, arma::vec());
}


const arma::vec& LinearRegression::predict(const arma::mat &X) const {

    return X * this -> theta;

}

const arma::vec& LinearRegression::get_theta() const {
    return this -> theta;
}





