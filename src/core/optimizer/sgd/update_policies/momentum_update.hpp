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

    double alpha_;
    double mu_;

public:
    MomentumUpdate(const double alpha, 
        const double mu): alpha_(alpha), mu_(mu) {};

    ~MomentumUpdate() {}; 

    /**
     * SGD with Momentum optimization method
    */
    void update(const VecType& W, const VecType& grad, VecType& updated_W) {
        std::size_t num_rows = W.rows();
        VecType V(num_rows);
        V.setZero();
        V = mu_ * V + alpha_ * grad;
        updated_W = W - V;
    };

};

}
}
#endif /*CORE_OPTIMIZER_SGD_UPDATE_POLICIES_MOMENTUM_UPDATE_HPP*/
