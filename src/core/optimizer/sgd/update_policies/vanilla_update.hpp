#ifndef CORE_OPTIMIZER_SGD_UPDATE_POLICIES_VANILLA_UPDATE_HPP
#define CORE_OPTIMIZER_SGD_UPDATE_POLICIES_VANILLA_UPDATE_HPP
#include "../../../../prereqs.hpp"
#include "../../../../core.hpp"

namespace openml {
namespace optimizer {

template<typename DataType>
class VanillaUpdate {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    double alpha;
public:

    VanillaUpdate(const double alpha_): alpha(alpha_) {};
    ~VanillaUpdate() {}; 

    void update(const VecType& W, const VecType& grad, VecType& updated_W) {
        updated_W = W - alpha * grad;
    };

};

}
}
#endif /*CORE_OPTIMIZER_SGD_UPDATE_POLICIES_VANILLA_UPDATE_HPP*/
