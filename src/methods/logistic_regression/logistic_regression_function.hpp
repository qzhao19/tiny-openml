#ifndef METHOD_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
#define METHOD_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_FUNCTION_HPP
#include "../prereqs.hpp"


namespace regression {


class LogisticRegressionFunction {
public:

    /**
     * create the logisticRegressionFunction constructor
     * 
     * @param X_, the matrix of input dataset
     * @param y_, the vector of label 
     * @param lamnda_, regularization coefficient  
    */
    
    LogisticRegressionFunction(const arma::mat &X_, 
                               const arma::vec &y_, 
                               const double lambda_);
    
    /**
     * deconstructor
    */
    ~LogisticRegressionFunction() {};

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


    double Evaluate(const arma::mat &theta, 
                    const size_t begin, 
                    const size_t batch_size) const;
    
    void Gradient(const arma::mat &theta, 
                  arma::mat &grad) const;


private:
    const arma::mat &X;
    const arma::vec &y;
    const double lambda;
    


};



};

#endif