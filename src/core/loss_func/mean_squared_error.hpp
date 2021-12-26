#ifndef CORE_LOSS_MEAN_SQUARED_ERROR_HPP
#define CORE_LOSS_MEAN_SQUARED_ERROR_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace math;

namespace loss {

class MeanSquaredError {
public:
    MeanSquaredError(
        const arma::mat &X_, 
        const arma::vec &y_) : X(X_), y(y_) {};

    ~MeanSquaredError() {};

    /**
     * shuffle dataset and the corresponding label, when we apply SGD 
    */
    void Shuffle() {
        arma::mat output_X;
        arma::vec output_y;

        math::shuffle_data(X, y, output_X, output_y);
    }

    /**
     * The MSE loss function MSE loss functionj: L(Y, Y_hat) = ||X * W - Y_hat||**2
     * @param W Matrix of shape [n_features, 1] 
     * @param begin Index of the starting point for loss function
     * @param batch_size Number of sample that will be passed 
    */
    double Evaluate(const arma::mat& W, 
        const std::size_t begin, 
        const std::size_t batch_size) {

        arma::mat X_batch = X.rows(begin, begin + batch_size - 1);
        arma::vec y_batch = y.rows(begin, begin + batch_size - 1);

        double retval = arma::accu(X_batch * W - y_batch);
    };

    /**
     * Calculate the gradient of MSE loss function with the given parameters
     * for the given batch size from a given index in the dataset
     * 
     *      gradients = 2 * X.T * X * W - 2 * X.T * Y
     * 
     * @param W Matrix of shape [n_features, 1] 
     * @param begin Index of the starting point for loss function
     * @param batch_size Number of sample that will be passed 
    */
    void Gradient(const arma::mat &W,  
        const std::size_t begin, 
        arma::mat &grads,
        const std::size_t batch_size) {
        
        arma::mat X_batch = X.rows(begin, begin + batch_size - 1);
        arma::vec y_batch = y.rows(begin, begin + batch_size - 1);

        grads = 2 * arma::trans(X) * X * W - 2 * arma::trans(X) * y;
        
    };
    
    std::size_t NumFunctions() const { return X.n_cols; }

private:
    const arma::mat &X;
    const arma::vec &y;

};

};
#endif
