#ifndef CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BACKTRACKING_HPP
#define CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BACKTRACKING_HPP
#include "../../../../prereqs.hpp"
#include "../../../../core.hpp"
#include "./base.hpp"

namespace openml {
namespace optimizer {

template <typename DataType,
    typename LossFunctionType,
    typename LineSearchParamType>
class LineSearchBacktracking: public BaseLineSearch<DataType, 
    LossFunctionType, 
    LineSearchParamType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

public:
    LineSearchBacktracking(const MatType& X, 
        const VecType& y,
        const LossFunctionType& loss_func,
        const LineSearchParamType& linesearch_params): BaseLineSearch<DataType, 
            LossFunctionType, 
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
        
        double dec_factor = this->linesearch_params_.dec_factor_;
        double inc_factor = this->linesearch_params_.inc_factor_;

        if (step <= 0.0) {
            std::cout << "ERROR: 'step' must be positive" << std::endl;
            return -1;
        }

        const double fx_init = fx;
        const double dg_init = d.dot(g);

        if (dg_init > 0.0) {
            std::cout << "Moving direction increases the objective function value" << std::endl;
            return -1;
        }

        double dg_test = this->linesearch_params_.ftol_ * dg_init;
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
                if (this->linesearch_params_.condition_ == "ARMIJO") {
                    return count;
                }
                // compute the project of d on the direction d
                double dg = d.dot(g);
                if (dg < this->linesearch_params_.wolfe_ * dg_init) {
                    width = inc_factor;
                }
                else {
                    if (this->linesearch_params_.condition_ == "WOLFE") {
                        return count;
                    }

                    if (dg > (-this->linesearch_params_.wolfe_ * dg_init)) {
                        width = dec_factor;
                    }
                    else {
                        return count;
                    }
                }
            }

            if (step < this->linesearch_params_.min_step_) {
                std::cout << "ERROR: the line search step became smaller than the minimum value allowed." << std::endl;
                return -1;
            }

            if (step > this->linesearch_params_.max_step_) {
                std::cout << "ERROR: the line search step became larger than the maximum value allowed." << std::endl;
                return -1;
            }

            if (count >= this->linesearch_params_.max_linesearch_) {
                std::cout << "ERROR: the line search step reached the max number of iterations." << std::endl;
                return -1;
            }
            step *= width;
        }
    }

};

}
}

#endif /* CORE_OPTIMIZER_LBFGS_SEARCH_POLICIES_BACKTRACKING_HPP */