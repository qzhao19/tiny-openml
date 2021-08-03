#ifndef METHOD_PERCEPTRON_PERCEPTRON_HPP
#define METHOD_PERCEPTRON_PERCEPTRON_HPP
#include "../../core.hpp"
#include "../../prereqs.hpp"
#include "weight_initializer/zero_initializer.hpp"


namespace perceptron {

template<typename WeightInitializer>
class Perceptron {

public:
    Perceptron(): initializer("zeros"), 
        shuffle(true),
        alpha(0.01), 
        tol(1e-5), 
        max_iter(10000) {};
    
    Perceptron(const std::string initializer_,
        const bool shuffle_,
        const double alpha_, 
        const double tol_, 
        const std::size_t max_iter_): initializer(initializer_), 
            shuffle(shuffle_),
            alpha(alpha_), 
            tol(tol_), 
            max_iter(max_iter_) {};

protected:
    void fit(const arma::mat &X, 
             const arma::vec &y) const;

    double sign(const arma::vec& x, 
                const arma::vec& w, 
                const double b) const;

private:
    arma::vec weights;
    double bias;

    std::string initializer;
    bool shuffle;
    double alpha;
    double tol;
    std::size_t max_iter;




};

};
#endif
