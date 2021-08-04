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

    for (std::size_t iter = 0; iter < max_iter; iter++) {

        arma::mat X_ = X;
        arma::vec y_ = y;

        if (shuffle) {
            shuffle_data(X, y, X_, y_);
        }

        for (std::size_t i = 0; i < n_samples; i++) {
            double total_error = 0.0;
            X_row = X_.row(i);
            y_row = y_(i);
            double y_pred = sign(X_row, weights, bias);
            total_error += std::pow(y_row - y_pred, 2); 
            if (y_row * y_pred)


        }


    }



}



