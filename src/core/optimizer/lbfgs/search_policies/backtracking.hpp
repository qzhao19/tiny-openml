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
        
        
        double dec_factor = this->linesearch_params_.decrease_factor;
        double inc_factor = this->linesearch_params_.increase_factor;

        if (step <= 0.0) {
            std::cout << "'step' must be positive" << std::endl;
            return -1;
        }

        const double fx_init = fx;
        const double dg_init = d.dot(g);

        if (dg_init > 0) {
            std::cout << "Moving direction increases the objective function value" << std::endl;
            return -1;
        }

        double dg_test = this->linesearch_params_.ftol * dg_init;
        double width;
        int count = 0;

        while (true) {
            // x_{k+1} = x_k + step * d_k
            x.noalias() = xp + step * d;

            fx = this->loss_func_.evaluate(this->X_, this->y_, x);
            g = this->loss_func_.gradient(this->X_, this->y_, x);

            ++count;

            if (fx > fx_init + step * dg_test) {
                width = dec_factor;
            }
            else {
                // check Armijo condition
                if (this->linesearch_params_.condition == "ARMIJO") {
                    return 1;
                }





            }
        }


    }

};

}
}

#endif /* CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BACKTRACKING_HPP */