#ifndef CORE_INITIALIZER_ZERO_INITIALIZER_HPP
#define CORE_INITIALIZER_ZERO_INITIALIZER_HPP
#include "../../prereqs.hpp"

namespace initializer {

class ZeroInitializer {
public:
    ZeroInitializer() {};

    inline static void Initialize(arma::vec& weights, 
        double& bias, 
        const std::size_t n_features) {
        
        weights.zeros(n_features);
        bias = 0.0;
    };

};

};

#endif
