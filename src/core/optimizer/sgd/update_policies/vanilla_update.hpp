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

    double alpha_;

public:

    VanillaUpdate(const double alpha): alpha_(alpha) {};

    ~VanillaUpdate() {}; 

    void update(const VecType& W, const VecType& grad, VecType& updated_W) {
        updated_W = W - alpha_ * grad;
    };

};

}
}
#endif /*CORE_OPTIMIZER_SGD_UPDATE_POLICIES_VANILLA_UPDATE_HPP*/
