#include "perceptron.hpp" 
using namespace perceptron;
using namespace math;

template<typename WeightInitializer>
double Perceptron<WeightInitializer>::sign(const arma::vec& x, 
    const arma::vec& w, 
    const double b) const {

    double y = arma::dot(x, w) + b;

    return (y >= 0.0) ? 1.0 : 0.0;
}


template<typename WeightInitializer>
void Perceptron<WeightInitializer>::fit(const arma::mat& X, 
    const arma::vec& y) const {
    
    // get number of cols == n_features and number of samples
    std::size_t n_samples = X.n_rows;
    std::size_t n_features = X.n_cols;

    WeightInitializer weight_initializer;

    weight_initializer.Initialize(weights, bias, n_features);

    std::size_t iter = 0;

    arma::mat X_shuffled = X;
    arma::vec y_shuffled = y;

    bool converged = false;

    while ((iter < max_iter) && (!converged)) {

        int error_count = 0;
        // shuffle dataset and associated label
        if (shuffle) {
            shuffle_data(X, y, X_shuffled, y_shuffled);
        }

        for (std::size_t i = 0; i < n_samples; i++) {
           
            X_ = X_shuffled.row(i);
            y_ = y_shuffled(i);

            double y_pred = sign(X_, weights, bias);
            if ((y_ * y_pred) <= 0.0) {
                weights = weights + alpha * X_ * y_;
                bias = bias + alpha * y_;
                error_count++;
            }
        }

        if (error_count == 0) {
            converged = true;
        }
    }

    this -> weights = weights;
    this -> bias = bias
}

