#ifndef CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BASE_HPP
#define CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BASE_HPP
#include "../../../../prereqs.hpp"
#include "../../../../core.hpp"

namespace openml {
namespace optimizer {

template <typename DataType>
class BaseLineSearch {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;



};

}
}

#endif /* CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BASE_HPP */