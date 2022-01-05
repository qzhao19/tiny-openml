#include "sgd_regressor.hpp"
using namespace regression;
using namespace initializer;
using namespace loss;

template<class LossFunctionType>
void SGDRegressor::fit(const arma::mat& X, 
    const arma::vec& y, 
    LossFunctionType& loss_fn) {
    
    std::size_t n_samples = X.n_rows;
    
    // loss_func.Gradient()
    // loss_fn_type loss_fn(X, y, lambda);
    arma::mat W_;

    // arma::mat X_ = X;
    // arma::vec y_ = y;

    for (std::size_t i = 0; i < max_iter; i++) {
        
        if (shuffle) {
            // math::shuffle_data(X, y, X_, y_);
            loss_fn.Shuffle();
        }

        std::size_t n_iter = n_samples / batch_size;

        for (std::size_t j = 0; j < n_iter; j++) {
            std::size_t begin = j * batch_size;
            arma::mat grads;

            loss_fn.Gradient(W_, begin, grads, batch_size);

            double learning_rate = 5 / (std::static_cast<double>(i * n_samples + j) + 50)

            // if (penalty.compare("l2")) {
            //     grads += lambda * W_;
            // }
            // else if (penalty.compare("l1")){
            //     grads += lambda * arma::sign(W_);
            // }
            // else if (penalty.compare("elasticnet")) {
            //     grads += lambda * l1_ratio * arma::sign(W_) +
            //         lambda * (1. - l1_ratio) * W_;
            // }
        }
    }

    W = arma::conv_to<arma::vec>(W_);
}


void SGDRegressor::fit(const arma::mat& X, const arma::vec& y) {

    MeanSquaredError mse(X, y, lambda);
    fit(X, y, mse);

}
