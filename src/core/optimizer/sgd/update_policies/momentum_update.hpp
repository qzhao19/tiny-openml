#ifndef CORE_OPTIMIZER_SGD_UPDATE_POLICIES_MOMENTUM_UPDATE_HPP
#define CORE_OPTIMIZER_SGD_UPDATE_POLICIES_MOMENTUM_UPDATE_HPP
#include "../../../../prereqs.hpp"
#include "../../../../core.hpp"

namespace openml {
namespace optimizer {

template<typename DataType>
class MomentumUpdate {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    double alpha;
    double mu;

public:

    MomentumUpdate(const double alpha_, 
        const double mu_): alpha(alpha_), mu(mu_) {};

    ~MomentumUpdate() {}; 

    /**
     * SGD with Momentum optimization method
    */
    void update(const VecType& W, const VecType& grad, VecType& updated_W) {
        std::size_t num_rows = W.rows();
        VecType V(num_rows);
        V.setZero();
        V = mu * V + alpha * grad;
        updated_W = W - V;
    };

};

}
}
#endif /*CORE_OPTIMIZER_SGD_UPDATE_POLICIES_MOMENTUM_UPDATE_HPP*/
