#ifndef CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BACKTRACKING_HPP
#define CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BACKTRACKING_HPP
#include "../../../../prereqs.hpp"
#include "../../../../core.hpp"
#include "./base.hpp"

namespace openml {
namespace optimizer {

template <typename DataType,
    typename LossFuncionType,
    typename LineSearchParamType>
class LineSearchBacktracking: public LineSearch<DataType, 
    LossFuncionType, 
    LineSearchParamType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

public:
    LineSearchBacktracking(const MatType& X, 
        const VecType& y,
        const LossFuncionType& loss_func,
        const LineSearchParamType& linesearch_params): LineSearch<DataType, 
            LossFuncionType, 
            LineSearchParamType>(
                X, y, 
                loss_func, 
                linesearch_params
        ) {};
    
    ~LineSearchBacktracking() {};

    int search(VecType& x, 
        double& fx, 
        VecType& g, 
        VecType& d, 
        double& step, 
        const VecType& xp, 
        const VecType& gp) {
        

    }

};

}
}

#endif /* CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BACKTRACKING_HPP */