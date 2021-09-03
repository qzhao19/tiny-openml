#ifndef METHOD_PERCEPTRON_WEIGHT_INITIALIZER_RAND_INITIALIZER_HPP
#define METHOD_PERCEPTRON_WEIGHT_INITIALIZER_RAND_INITIALIZER_HPP
#include "../../../prereqs.hpp"

namespace perceptron {

class RandInitializer {
public:
    RandInitializer() {};

    void Initialize(arma::vec& weights, 
        double& bias, 
        const std::size_t n_features) {
        
        arma::vec w(n_features, arma::fill::randu);
        weights = w;

        double b = (double)rand() / (double)(RAND_MAX / 1.0);
        bias = b;
    };
};

};

#endif
