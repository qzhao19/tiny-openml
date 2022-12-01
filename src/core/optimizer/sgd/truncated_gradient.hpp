#ifndef CORE_OPTIMIZER_SGD_TRUNCATED_GRADIENT_HPP
#define CORE_OPTIMIZER_SGD_TRUNCATED_GRADIENT_HPP
#include "./decay_policies/step_decay.hpp"
#include "./decay_policies/exponential_decay.hpp"
#include "./update_policies/vanilla_update.hpp"
#include "./update_policies/momentum_update.hpp"
#include "../../../prereqs.hpp"
#include "../../../core.hpp"
using namespace openml;

namespace openml {
namespace optimizer {

template<typename DataType, 
    typename LossFuncionType, 
    typename UpdatePolicyType,
    typename DecayPolicyType>
class TruncatedGradient: public BaseOptimizer<DataType, 
    LossFuncionType, 
    UpdatePolicyType, 
    DecayPolicyType> {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    double alpha_;
    double l1_ratio_;
   
    /**
     * truncated gradient implementation
    */
    void truncate(VecType& weight, 
        VecType& cum_l1,
        DataType max_cum_l1) {
        
        std::size_t num_features = weight.rows();
        
        for (std::size_t j = 0; j < num_features; ++j) {
            DataType w_j = weight(j, 0);
            if (w_j > 0.0) {
                weight(j, 0) = std::max(0.0, w_j - (max_cum_l1 + cum_l1(j, 0)));
            }
            else if (w_j < 0.0) {
                weight(j, 0) = std::min(0.0, w_j + (max_cum_l1 - cum_l1(j, 0)));
            }

            cum_l1(j, 0) += (weight(j, 0) - w_j);
        }
        cum_l1.setZero();
    }

public:
    TruncatedGradient(const VecType& x0,
        const LossFuncionType& loss_func,
        const UpdatePolicyType& w_update,
        const DecayPolicyType& lr_decay,
        const std::size_t max_iter = 2000, 
        const std::size_t batch_size = 20,
        const std::size_t num_iters_no_change = 5,
        const double tol = 0.0001, 
        const double alpha = 0.0001,
        const double l1_ratio = 0.15,
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
                verbose),
            alpha_(alpha),
            l1_ratio_(l1_ratio) {};
    ~TruncatedGradient() {};

    
    void optimize(const MatType& X, 
        const VecType& y) {
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        std::size_t num_batch = num_samples / this->batch_size_;
        std::size_t no_improvement_count = 0;

        bool is_converged = false;
        double best_loss = ConstType<double>::infinity();
        DataType max_cum_l1 = 0.0;
        
        MatType X_new = X;
        VecType y_new = y;
        VecType cum_l1(num_features);
        cum_l1.setZero();
        for (std::size_t iter = 0; iter < this->max_iter_; iter++) {
            if (this->shuffle_) {
                random::shuffle_data<MatType, VecType>(X_new, y_new, X_new, y_new);
            }
            MatType X_batch(this->batch_size_, num_features);
            VecType y_batch(this->batch_size_);
            VecType loss_history(this->batch_size_);

            double lr = lr_decay_.compute(iter);

            for (std::size_t j = 0; j < num_batch; j++) {
                std::size_t begin = j * this->batch_size_;
                X_batch = X_new.middleRows(begin, this->batch_size_);
                y_batch = y_new.middleRows(begin, this->batch_size_);

                VecType grad(num_features);
                grad = loss_func_.gradient(X_batch, y_batch, this->x0_);

                // clip gradient with large value 
                grad = utils::clip<MatType>(grad, this->MAX_DLOSS, this->MIN_DLOSS);

                // W = W - lr * grad; 
                this->x0_ = w_update_.update(this->x0_, grad, lr);

                max_cum_l1 += static_cast<DataType>(l1_ratio_) * 
                    static_cast<DataType>(lr) * static_cast<DataType>(alpha_);
                truncate(this->x0_, cum_l1, max_cum_l1);

                double loss = loss_func_.evaluate(X_batch, y_batch, this->x0_);

                loss_history(j, 0) = loss;
            }

            double sum_loss = static_cast<double>(loss_history.array().sum());

            if (sum_loss > best_loss - this->tol_ * this->batch_size_) {
                no_improvement_count +=1;
            }
            else {
                no_improvement_count = 0;
            }

            if (sum_loss < best_loss) {
                best_loss = sum_loss;
            }

            if (no_improvement_count >= this->num_iters_no_change_) {
                is_converged = true;
                this->opt_x_ = this->x0_;
                break;
            }

            if (this->verbose_) {
                if ((iter % 2) == 0) {
                    std::cout << "-- Epoch = " << iter << ", average loss value = " 
                              << sum_loss / static_cast<double>(this->batch_size_) << std::endl;
                }
            }
        }

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
#endif /*CORE_OPTIMIZER_SGD_TRUNCATED_GRADIENT_HPP*/
