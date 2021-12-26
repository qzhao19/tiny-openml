#ifndef CORE_LOSS_FUNC_LOG_LOSS_HPP
#define CORE_LOSS_FUNC_LOG_LOSS_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace math;

namespace loss {

class LogLoss {
public:
    /**
     * create the logisticRegressionFunction constructor
     * 
     * @param X_, the matrix of input dataset
     * @param y_, the vector of label 
     * @param lambda_, regularization coefficient  
    */
    LogLoss(const arma::mat &X_, 
        const arma::vec &y_, 
        const double lambda_): X(X_), y(y_), lambda(lambda_) { };
    
    /**
     * deconstructor
    */
    ~LogLoss() {};


    /**
     * shuffle the dataset and associated label, when we apply the SGD method.
     * This may be called by the optimizer.
    */
    void Shuffle() {
        arma::mat output_X;
        arma::vec output_y;

        math::shuffle_data(X, y, output_X, output_y);
    };

    /**
     * evaluate the gradient of the logistic regression log-likelihood function
     * with the given parameters. f a point has 0 probability of being classified
     * directly with the given parameters, then Evaluate() will return nan (this
     * is kind of a corner case and should not happen for reasonable models).
     * The optiumn of this function is 0.0
     * 
     * @param W, the vector of logistic regression parameters 
    */
    double Evaluate(const arma::mat &W) const {
        /**
         * The objective function is likelihood function, W is the vector 
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
        const double penalty = (lambda / (2.0 * n_features)) * arma::dot(W, W);

        // define the sigmoid function h(x) = 1 / (1 + exp(w'x))
        const arma::vec sigmoid = 1.0 / (1.0 + arma::exp(-1 * X * W));

        // the objective function we want to minimize, so we make positive function as negative
        // -1 / n_features 
        const double cost_fn = -0.5 * (arma::dot(y, arma::log(sigmoid)) + 
            arma::dot(1 - y, arma::log(1 - sigmoid)));

        return cost_fn + penalty;
    };

    /**
     * Evaluate the gradient of the logistic regression log-likelihood function
     * with the given parameters.
     * 
     * @param W Vector of logistic regression parameters
     * @param grads Vector of output gradient
     * 
    */
    void Gradient(const arma::mat &W, 
        arma::mat &grads) const {
        /**
         * Evaluate the gradient of the logistic regression objective function.
         *      grad = (1 / m) * sum(sigmoid(x) - y) + (lambda / m) * W
        */

        // get the number of features ==> nb of cols
        const int n_features = X.n_cols;

        // define the sigmoid function h(x) = 1 / (1 + exp(w'x))
        const arma::vec sigmoid = 1.0 / (1.0 + arma::exp(-1 * X * W));

        arma::mat penalty = lambda * W / n_features;

        grads.set_size(arma::size(W));

        grads = X.t() * (sigmoid - y) + penalty;
    };


    /**
     * Evaluate the gradient of logistic regression log-likelihood function with the given 
     * parameters using the given batch size from the given point index. This is specificly
     * for SGD-like optimizers, which require a seperable objective functions  
     * 
     * @param W Vector of logistic regression parameters 
     * @param begin Index of the starting point for loss function
     * @param batch_size Number of sample that will be passed 
    */
    double Evaluate(const arma::mat &W, 
        const std::size_t begin, 
        const std::size_t batch_size) const {
        /**
         * define the objective function f(x)
         * 
         *   f(x) = sum(y_batch * log(sigmoid(W * X_batch)) + (1 - y_batch) * log(1 - sigmoid(W * X_batch)))
        */
        const int n_features = X.n_cols;

        // regularization term penalty = lambda * batch_size / 2 * m * sum(W**2), here m = n_features
        const double penalty = ((lambda * batch_size) / (2.0 * n_features)) * 
            arma::dot(W, W);


        // define dataset of one batch and vector of label associated
        arma::mat X_batch = X.rows(begin, begin + batch_size - 1);
        arma::vec y_batch = y.rows(begin, begin + batch_size - 1);

        // define the sigmoid function 
        // sigmoid(z) = 1 / (1 + exp(-z))
        const arma::vec sigmoid = 1.0 / (1.0 + arma::exp(-1 * X_batch * W));

        // objective function
        const double cost_fn = (-1.0 / 2.0) * 
            (arma::dot(y_batch, arma::log(sigmoid)) + 
                arma::dot((1 - y_batch), arma::log(1 - sigmoid)));
        
        return cost_fn + penalty;
    };
    
    /**
     * Calculate the gradient of logistic regression function with the given parameters
     * for the given batch size from a given index in the dataset
    */
    void Gradient(const arma::mat &W, 
        const std::size_t begin,
        arma::mat &grads,  
        const std::size_t batch_size) const {
        /**
         * calculate gradient vector 
         *      gradient = (1 / m) * sum(sigmoid(x) - y) + (lambda / m) * W
        */
        const int n_features = X.n_cols;

        arma::mat penalty = lambda * W / n_features * batch_size;

        // define dataset of one batch and vector of label associated
        arma::mat X_batch = X.rows(begin, begin + batch_size - 1);
        arma::vec y_batch = y.rows(begin, begin + batch_size - 1);

        const arma::vec sigmoid = 1.0 / (1.0 + arma::exp(-1 * X_batch * W));

        grads.set_size(arma::size(W));

        grads = X_batch.t() * (sigmoid - y_batch) + penalty;
    }
    
    //! Return the number of separable functions (the number of predictor points).
    std::size_t NumFunctions() const { return X.n_cols; }

private:
    const arma::mat &X;
    const arma::vec &y;
    double lambda;
    
};

};
#endif
