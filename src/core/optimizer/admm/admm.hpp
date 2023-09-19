#ifndef CORE_OPTIMIZER_ADMM_ADMM_HPP
#define CORE_OPTIMIZER_ADMM_ADMM_HPP
#include "../../../prereqs.hpp"
#include "../../../core.hpp"
#include "../base.hpp"
using namespace openml;

namespace openml {
namespace optimizer {

template<typename DataType, 
    typename LossFuncionType, 
    typename UpdatePolicyType,
    typename DecayPolicyType>
class ADMM: public BaseOptimizer<DataType, 
    LossFuncionType, 
    UpdatePolicyType, 
    DecayPolicyType> {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

public:
    
};

}
}
#endif /*CORE_OPTIMIZER_ADMM_ADMM_HPP*/
