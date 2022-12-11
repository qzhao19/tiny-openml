#ifndef CORE_OPTIMIZER_BASE_HPP
#define CORE_OPTIMIZER_BASE_HPP
#include "./sgd/decay_policies/step_decay.hpp"
#include "./sgd/decay_policies/exponential_decay.hpp"
#include "./sgd/update_policies/vanilla_update.hpp"
#include "./sgd/update_policies/momentum_update.hpp"
#include "../../prereqs.hpp"
#include "../../core.hpp"

namespace openml {
namespace optimizer {

template<typename DataType, 
    typename LossFuncionType, 
    typename UpdatePolicyType = optimizer::VanillaUpdate<DataType>,
    typename DecayPolicyType = optimizer::StepDecay<DataType>>
class BaseOptimizer {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

protected:
    LossFuncionType loss_func_;
    UpdatePolicyType w_update_;
    DecayPolicyType lr_decay_;

    std::size_t max_iter_;
    std::size_t batch_size_;
    std::size_t num_iters_no_change_;
    double tol_;
    bool shuffle_;
    bool verbose_;

    VecType x0_;
    VecType opt_x_;
    // internal callable parameters
    DataType MAX_DLOSS = static_cast<DataType>(1e+10);
    DataType MIN_DLOSS = -static_cast<DataType>(1e+10);

public:
    BaseOptimizer(const VecType& x0,
        const LossFuncionType& loss_func,
        const UpdatePolicyType& w_update,
        const DecayPolicyType& lr_decay,
        const std::size_t max_iter = 2000,
        const std::size_t batch_size = 16,
        const std::size_t num_iters_no_change = 5,
        const double tol = 0.0001, 
        const bool shuffle = true, 
        const bool verbose = true): x0_(x0),
            loss_func_(loss_func),
            w_update_(w_update),
            lr_decay_(lr_decay),
            max_iter_(max_iter), 
            batch_size_(batch_size),
            num_iters_no_change_(num_iters_no_change),
            tol_(tol), 
            shuffle_(shuffle),
            verbose_(verbose) {};
    
    BaseOptimizer(const VecType& x0,
        const LossFuncionType& loss_func,
        const std::size_t max_iter = 2000,
        const bool shuffle = true, 
        const bool verbose = true): x0_(x0),
            loss_func_(loss_func),
            max_iter_(max_iter), 
            shuffle_(shuffle),
            verbose_(verbose) {};
    ~BaseOptimizer() {};

    virtual void optimize(const MatType& X, 
        const VecType& y) = 0;

    const VecType get_coef() const {
        return opt_x_;
    }
};

}
}
#endif /*CORE_OPTIMIZER_BASE_HPP*/