#ifndef CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BASE_HPP
#define CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BASE_HPP
#include "../../../../prereqs.hpp"
#include "../../../../core.hpp"

namespace openml {
namespace optimizer {

template <typename DataType,
    typename LossFuncionType,
    typename LineSearchParamType>
class LineSearch {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

protected:
    MatType X_;
    VecType y_;
    LossFuncionType loss_func_;
    LineSearchParamType linesearch_params_;

public:
    LineSearch() {};
    LineSearch(const MatType& X, 
        const VecType& y,
        const LossFuncionType& loss_func,
        const LineSearchParamType& linesearch_params): X_(X), 
            y_(y), 
            loss_func_(loss_func), 
            linesearch_params_(linesearch_params) {};

    ~LineSearch() {};

    virtual int search(VecType& x, 
        double& fx, 
        VecType& g, 
        VecType& d, 
        double& step, 
        const VecType& xp, 
        const VecType& gp) = 0;

};

}
}
#endif /* CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BASE_HPP */