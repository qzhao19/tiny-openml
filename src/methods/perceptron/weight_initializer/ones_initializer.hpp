#ifndef METHOD_PERCEPTRON_WEIGHT_INITIALIZER_ONES_INITIALIZER_HPP
#define METHOD_PERCEPTRON_WEIGHT_INITIALIZER_ONES_INITIALIZER_HPP
#include "../../../prereqs.hpp"

namespace perceptron {

class OnesInitializer {
public:
    OnesInitializer() {};

    inline static void Initialize(arma::rowvec& weights, 
        double bias, 
        const std::size_t n_features) {
        
        weights.ones(n_features);
        bias = 0.0;
    };

};

};

#endif
