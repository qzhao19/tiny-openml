
#include "linear_regression.hpp"
using namespace regression;

void LinearRegression::fit_(const arma::mat &X, 
                            const arma::vec &y, 
                            const arma::vec &weights) {
    /**
     * Calculate the closed form solution
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
    if (penalty.empty()) {
        arma::mat pseudo_inv = arma::inv(X_.t() * X_);
        W = pseudo_inv * X_.t() * y_;
    }
    else {
        arma::mat pseudo_inv = arma::inv(X_.t() * X_ + this -> lambda * identifty_mat);
        W = pseudo_inv * X_.t() * y_;
    }
    // W = arma::inv(pseudo_inv) * X_.t() * y_;
};

void LinearRegression::fit(const arma::mat &X, 
                           const arma::vec &y, 
                           const arma::vec &weights) {
    fit_(X, y, weights);
}

void LinearRegression::fit(const arma::mat &X, 
                           const arma::vec &y) {
    fit_(X, y, arma::vec());
}

const arma::vec LinearRegression::predict(const arma::mat &X) const {
    /**
     * y_pred = X * theta
    */
    arma::vec y_pred;
    
    if (intercept) {
        // if intercept, need split the intercept term from theta vector
        y_pred = X * W.subvec(0, W.n_rows - 2);
        y_pred += W(W.n_rows - 1);
        return y_pred;
    } 
    else {
        y_pred = X * W;
        return y_pred;
    }
}

const double LinearRegression::score(const arma::vec &y_true, 
                                     const arma::vec &y_pred) const {
    /**
     * Return the coeffiecient of determination R^2
     * R^2 = (1 - u / v), u = ((y_pred - y_true) ** 2).sum()
     * v = ((y_true - y_true.mean()) ** 2).sum()
    */
   double u = arma::sum(arma::square(y_pred - y_true));
   double v = arma::sum(arma::square(y_true - arma::mean(y_true)));
   return 1 - u / v;

}

const arma::vec& LinearRegression::get_coef() const {
    return this -> W;
}
