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
    const VecType update(const VecType& W, const VecType& grad, const double lr) const {
        std::size_t num_rows = W.rows();
        VecType V(num_rows);
        V.setZero();
        
        VecType prev_V(num_rows);
        prev_V = V;

        V = mu * V + static_cast<DataType>(lr) * grad;
        V = V + mu * (V - prev_V);
        VecType updated_W = W - V;
        return updated_W;
    };

};

}
}
#endif /*CORE_OPTIMIZER_SGD_UPDATE_POLICIES_NESTEROV_MOMENTUM_UPDATE_HPP*/