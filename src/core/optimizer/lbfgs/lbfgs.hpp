#ifndef CORE_OPTIMIZER_LBFGS_LBFGS_HPP
#define CORE_OPTIMIZER_LBFGS_LBFGS_HPP
#include "../../../prereqs.hpp"
#include "../../../core.hpp"
#include "../base.hpp"
using namespace openml;

namespace openml {
namespace optimizer {

/**
 * Stochastic coordinate descent algorithm
*/
template<typename DataType, 
    typename LossFuncionType,
    typename LineSearchParamType>
class LBFGS: public BaseOptimizer<DataType, 
    LossFuncionType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    std::string linesearch_policy_;
    std::size_t mem_size_;
    std::size_t past_;
    double tol_;
    double delta_; 
    LineSearchParamType linesearch_params_;

public:
    LBFGS(const VecType& x0,
        const LossFuncionType& loss_func,
        const LineSearchParamType& linesearch_params,
        const std::string linesearch_policy = "backtracking",
        const std::size_t max_iter = 0, 
        const std::size_t mem_size = 8, 
        const std::size_t past = 3,
        const double tol = 1e-5,
        const double delta = 1e-6,
        const bool shuffle = true,
        const bool verbose = true): BaseOptimizer<DataType, 
            LossFuncionType>(x0, 
                loss_func, 
                max_iter, 
                shuffle, 
                verbose),
            linesearch_params_(linesearch_params),
            mem_size_(mem_size),
            past_(past),
            tol_(tol),
            delta_(delta) {};
    
    ~LBFGS() {};

    void optimize(const MatType& X, 
        const VecType& y) {
        
        // the initial parameters and its size
        VecType x0 = this->x0;
        std::size_t num_dims = x0.rows();

        // intermediate variables: previous x, gradient, previous gradient, directions
        VecType xp(num_dims);
        VecType g(num_dims);
        VecType gp(num_dims);
        VecType d(num_dims);

        // an array for storing previous values of the objective function
        VecType fxp(past_);



    }

};

}
}
#endif /*CORE_OPTIMIZER_LBFGS_LBFGS_HPP*/
