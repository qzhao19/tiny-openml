#ifndef METHOD_PERCEPTRON_WEIGHT_INITIALIZER_ZERO_INITIALIZER_HPP
#define METHOD_PERCEPTRON_WEIGHT_INITIALIZER_ZERO_INITIALIZER_HPP
#include "../../../prereqs.hpp"

namespace perceptron {

class ZeroInitializer {
public:
    ZeroInitializer() {};

    inline static void Initialize(arma::vec& weights, 
        double bias, 
        const std::size_t n_features) {
        
        weights.zeros();
        bias = 0.0;
    };

};

};

#endif
