#include "logistic_regression_function.hpp"
using namespace regression;
using namespace math;

void LogisticRegressionFunction::Shuffle() {
    arma::mat output_X;
    arma::vec output_y;

    math::shuffle_data(X, y, output_X, output_y);
    
}


double LogisticRegressionFunction::Evaluate(const arma::mat &theta) const {
    /**
     * The objective function is likelihood function, theta is the vector 
     * of parameters, y is the result, X is the matrix of data
     * 
     * f(w) = sum(y_i * log(sigmoid(w.t() * x_i)) + (1 - y_i) * log(1 - sigmoid(w.t() * x_i)))
     * need to minimize this function.
     * 
     * L2-regularization term is the lambda multiply by the squared l2-norm then divide
     * by 2
     * 
    */

    // get the number of features ==> nb of cols
    const int n_features = X.n_cols;

    // the regularization term P(w) = lambda / m * norm2(w)
    const double penality = (lambda / (2.0 * n_features)) * arma::dot(theta, theta);

    // define the sigmoid function h(x) = 1 / (1 + exp(w'x))
    const arma::vec sigmoid = 1.0 / (1.0 + arma::exp(-1 * X * theta));

    // the objective function we want to minimize, so we make positive function as negative
    // -1 / n_features 
    const double cost_fn = -0.5 * (arma::dot(y, arma::log(sigmoid)) + 
        arma::dot(1 - y, arma::log(1 - sigmoid)));

    return cost_fn + penality;
}


void LogisticRegressionFunction::Gradient(const arma::mat &theta, 
                                          arma::mat &grads) const {
    /**
     * Evaluate the gradient of the logistic regression objective function.
     *      grad = (1 / m) * sum(sigmoid(x) - y) + (lambda / m) * theta
    */

    // get the number of features ==> nb of cols
    const int n_features = X.n_cols;

    // define the sigmoid function h(x) = 1 / (1 + exp(w'x))
    const arma::vec sigmoid = 1.0 / (1.0 + arma::exp(-1 * X * theta));

    arma::mat penality = lambda * theta / n_features;

    grads.set_size(arma::size(theta));

    grads = X.t() * (sigmoid - y) + penality;
}


double LogisticRegressionFunction::Evaluate(const arma::mat &theta, 
                                            const std::size_t begin, 
                                            const std::size_t batch_size) const {
    /**
     * define the objective function f(x)
     * 
     *   f(x) = sum(y_batch * log(sigmoid(W * X_batch)) + (1 - y_batch) * log(1 - sigmoid(W * X_batch)))
    */
    const int n_features = X.n_cols;

    // regularization term penality = lambda * batch_size / 2 * m * sum(W**2), here m = n_features
    const double penality = ((lambda * batch_size) / (2.0 * n_features)) * 
        arma::dot(theta, theta);


    // define dataset of one batch and vector of label associated
    arma::mat X_batch = X.rows(begin, begin + batch_size - 1);
    arma::vec y_batch = y.rows(begin, begin + batch_size - 1);

    // define the sigmoid function 
    // sigmoid(z) = 1 / (1 + exp(-z))
    const arma::vec sigmoid = 1.0 / (1.0 + arma::exp(-1 * X_batch * theta));

    // objective function
    const double cost_fn = (-1.0 / 2.0) * 
        (arma::dot(y_batch, arma::log(sigmoid)) + 
            arma::dot((1 - y_batch), arma::log(1 - sigmoid)));
    
    return cost_fn + penality;

}


void LogisticRegressionFunction::Gradient(const arma::mat &theta,  
                                          const std::size_t begin, 
                                          arma::mat &grads,
                                          const std::size_t batch_size) const {
    /**
     * calculate gradient vector 
     *      gradient = (1 / m) * sum(sigmoid(x) - y) + (lambda / m) * theta
    */
    const int n_features = X.n_cols;

    arma::mat penality = lambda * theta / n_features * batch_size;

    // define dataset of one batch and vector of label associated
    arma::mat X_batch = X.rows(begin, begin + batch_size - 1);
    arma::vec y_batch = y.rows(begin, begin + batch_size - 1);

    const arma::vec sigmoid = 1.0 / (1.0 + arma::exp(-1 * X_batch * theta));

    grads.set_size(arma::size(theta));

    grads = X_batch.t() * (sigmoid - y_batch) + penality;

}
