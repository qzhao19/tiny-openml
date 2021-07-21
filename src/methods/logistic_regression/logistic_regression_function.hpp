#ifndef METHOD_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
#define METHOD_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"


namespace regression {


class LogisticRegressionFunction {
public:

    /**
     * create the logisticRegressionFunction constructor
     * 
     * @param X_, the matrix of input dataset
     * @param y_, the vector of label 
     * @param lambda_, regularization coefficient  
    */
    
    LogisticRegressionFunction(
        const arma::mat &X_, 
        const arma::vec &y_, 
        const double lambda_): X(X_), y(y_), lambda(lambda_) { };
    
    /**
     * deconstructor
    */
    ~LogisticRegressionFunction() {};


    /**
     * shuffle the dataset and associated label, when we apply the SGD method.
     * This may be called by the optimizer.
    */
    void Shuffle();

    /**
     * evaluate the gradient of the logistic regression log-likelihood function
     * with the given parameters. f a point has 0 probability of being classified
     * directly with the given parameters, then Evaluate() will return nan (this
     * is kind of a corner case and should not happen for reasonable models).
     * The optiumn of this function is 0.0
     * 
     * @param theta, the vector of logistic regression parameters 
    */
    double Evaluate(const arma::mat &theta) const;

    /**
     * Evaluate the gradient of the logistic regression log-likelihood function
     * with the given parameters.
     * 
     * @param theta Vector of logistic regression parameters
     * @param grads Vector of output gradient
     * 
    */
    void Gradient(const arma::mat &theta, 
                  arma::mat &grads) const;


    /**
     * Evaluate the gradient of logistic regression log-likelihood function with the given 
     * parameters using the given batch size from the given point index. This is specificly
     * for SGD-like optimizers, which require a seperable objective functions  
     * 
     * @param theta Vector of logistic regression parameters 
     * @param begin Index of the starting point for loss function
     * @param batch_size Number of sample that will be passed 
    */
    double Evaluate(const arma::mat &theta, 
                    const std::size_t begin, 
                    const std::size_t batch_size) const;
    
    /**
     * Calculate the gradient of logistic regression function with the given parameters
     * for the given batch size from a given index in the dataset
    */
    void Gradient(const arma::mat &theta, 
                  const std::size_t begin,
                  arma::mat &grads,  
                  const std::size_t batch_size) const;
    
    //! Return the number of separable functions (the number of predictor points).
    std::size_t NumFunctions() const { return X.n_cols; }

private:
    const arma::mat &X;
    const arma::vec &y;
    double lambda;
    
};

};

#endif
