#ifndef METHOD_PERCEPTRON_PERCEPTRON_HPP
#define METHOD_PERCEPTRON_PERCEPTRON_HPP
#include "../../prereqs.hpp"



namespace perceptron {

class Perceptron {

public:
    Perceptron(const double alpha_, 
               const double tol_, 
               const std::size_t max_iter_): 
        alpha(alpha_), 
        tol(tol_), 
        max_iter(max_iter_) {};


    Perceptron(): alpha(0.01), 
                  tol(1e-5), 
                  max_iter(10000) {};
    

    


protected:
    void fit(const arma::mat &X, const arma::vec &y);


private:
    arma::mat weights;
    arma::vec biases;

    double alpha;
    double tol;
    std::size_t max_iter;




};

};
#endif
