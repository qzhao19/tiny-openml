#ifndef CORE_OPTIMIZER_LBFGS_LBFGS_HPP
#define CORE_OPTIMIZER_LBFGS_LBFGS_HPP
#include "../../../prereqs.hpp"
#include "../../../core.hpp"
using namespace openml;

namespace openml {
namespace optimizer {

/**
 * Stochastic coordinate descent algorithm
*/
template<typename DataType>
class LBFGS {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    std::size_t mem_size_;
    std::size_t past_;
    std::size_t max_iters;
    double tol_;
    double delta_; 

    





};

}
}
#endif /*CORE_OPTIMIZER_LBFGS_LBFGS_HPP*/
