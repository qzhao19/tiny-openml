#ifndef CORE_OPTIMIZER_SGD_UPDATE_POLICIES_NESTEROV_MOMENTUM_UPDATE_HPP
#define CORE_OPTIMIZER_SGD_UPDATE_POLICIES_NESTEROV_MOMENTUM_UPDATE_HPP
#include "../../../../prereqs.hpp"
#include "../../../../core.hpp"

namespace openml {
namespace optimizer {

template<typename DataType>
class NesterovMomentumUpdate {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    DataType mu;

public:
    NesterovMomentumUpdate(const DataType mu_): mu(mu_) {};
    ~NesterovMomentumUpdate() {}; 

    /**
     * SGD with Nesterov Momentum optimization method
    */
    const MatType update(const MatType& W, const MatType& grad, const double lr) const {
        std::size_t nrows = W.rows();
        MatType V(nrows);
        V.setZero();
        
        MatType prev_V(nrows);
        prev_V = V;

        V = mu * V + static_cast<DataType>(lr) * grad;
        V = V + mu * (V - prev_V);
        MatType updated_W = W - V;
        return updated_W;
    };

};

}
}
#endif /*CORE_OPTIMIZER_SGD_UPDATE_POLICIES_NESTEROV_MOMENTUM_UPDATE_HPP*/