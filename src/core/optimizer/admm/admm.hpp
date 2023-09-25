#ifndef CORE_OPTIMIZER_ADMM_ADMM_HPP
#define CORE_OPTIMIZER_ADMM_ADMM_HPP
#include "../../../prereqs.hpp"
#include "../../../core.hpp"
#include "../base.hpp"
using namespace openml;

namespace openml {
namespace optimizer {

template<typename DataType, 
    typename LossFuncionType, 
    typename UpdatePolicyType,
    typename DecayPolicyType>
class ADMM: public BaseOptimizer<DataType, 
    LossFuncionType, 
    UpdatePolicyType, 
    DecayPolicyType> {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

public:
    ADMM(const VecType& x0,
        const LossFuncionType& loss_func,
        const UpdatePolicyType& w_update,
        const DecayPolicyType& lr_decay,
        const std::size_t max_iter = 2000, 
        const std::size_t batch_size = 24,
        const std::size_t num_iters_no_change = 5,
        const double tol = 0.0001, 
        const bool shuffle = true, 
        const bool verbose = true): BaseOptimizer<DataType, 
            LossFuncionType, 
            UpdatePolicyType, 
            DecayPolicyType>(x0, 
                loss_func, 
                w_update, 
                lr_decay, 
                max_iter, 
                batch_size, 
                num_iters_no_change, 
                tol, 
                shuffle, 
                verbose) {};
    ~ADMM() {};

    void optimize(const MatType& X, 
        const VecType& y) {
        std::size_t num_samples = X.rows(), num_features = X.cols();
        std::size_t num_batch = num_samples / this->batch_size_;
        std::size_t no_improvement_count = 0;
        
        bool is_converged = false;
        double best_loss = ConstType<double>::infinity();

        if (!is_converged) {
            std::ostringstream err_msg;
            err_msg << "Not converge, current number of epoch = " << this->max_iter_
                    << ", the batch size = " << this->batch_size_ 
                    << ", try apply different parameters." << std::endl;
            throw std::runtime_error(err_msg.str());
        }
    }

};

}
}
#endif /*CORE_OPTIMIZER_ADMM_ADMM_HPP*/
