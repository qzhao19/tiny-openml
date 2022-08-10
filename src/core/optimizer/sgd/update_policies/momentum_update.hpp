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
    
    double mu_;

public:
    MomentumUpdate(const double mu): mu_(mu) {};
    ~MomentumUpdate() {}; 

    /**
     * SGD with Momentum optimization method
    */
    const VecType update(const VecType& W, 
        const VecType& grad, 
        const double lr) const{
        std::size_t num_rows = W.rows();
        VecType V(num_rows);
        V.setZero();
        V = mu_ * V + lr * grad;
        VecType updated_W = W - V;
        return updated_W;
    };

};

}
}
#endif /*CORE_OPTIMIZER_SGD_UPDATE_POLICIES_MOMENTUM_UPDATE_HPP*/
