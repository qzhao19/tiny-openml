#include "perceptron.hpp" 
using namespace perceptron;
using namespace math;

template<typename WeightInitializer>
double Perceptron<WeightInitializer>::sign(const arma::rowvec& x, 
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

    // define local weights and bias, and initialize them by
    // weight_initializer  
    arma::vec w;
    double b;

    WeightInitializer weight_initializer;
    weight_initializer.Initialize(w, b, n_features);

    arma::mat X_shuffled = X;
    arma::vec y_shuffled = y;

    std::size_t iter = 0;

    while (iter < max_iter) {
        // shuffle dataset and associated label
        if (shuffle) {
            math::shuffle_data(X, y, X_shuffled, y_shuffled);
        }

        for (std::size_t i = 0; i < n_samples; i++) {
            arma::rowvec X_row = X_shuffled.row(i);
            double y_row = y_shuffled(i);
            double y_pred = sign(X_row, w, b);

            if ((y_row * y_pred) <= 0.0) {
                arma::vec X_trans = arma::conv_to<arma::vec>::from(X_row);
                w = w + alpha * X_trans * y_row;
                b = b + alpha * y_row;
            }
        }
        iter++;
    }

    this -> weights = w;
    this -> bias = b;

}


template<typename WeightInitializer>
const arma::mat Perceptron<WeightInitializer>::predict(
    const arma::mat& X) const {

    arma::vec y_pred = X * weights + bias;
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


template<typename WeightInitializer>
const double Perceptron<WeightInitializer>::score(
    const arma::vec &y_true, 
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
