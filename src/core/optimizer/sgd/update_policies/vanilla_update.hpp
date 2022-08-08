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

public:
    VanillaUpdate() {};
    ~VanillaUpdate() {}; 

    const VecType update(double lr, const VecType& W, const VecType& grad) const {
        VecType updated_W = W - grad * static_cast<DataType>(lr);
        return updated_W;
    };

};

}
}
#endif /*CORE_OPTIMIZER_SGD_UPDATE_POLICIES_VANILLA_UPDATE_HPP*/
