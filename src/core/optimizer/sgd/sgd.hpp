#ifndef CORE_OPTIMIZER_SGD_SGD_HPP
#define CORE_OPTIMIZER_SGD_SGD_HPP
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
class SGD: public BaseOptimizer<DataType, 
    LossFuncionType, 
    UpdatePolicyType, 
    DecayPolicyType> {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

public:
    SGD(const MatType& x0,
        const LossFuncionType& loss_func,
        const UpdatePolicyType& w_update,
        const DecayPolicyType& lr_decay,
        const std::size_t max_iter = 2000,
        const std::size_t batch_size = 16,
        const std::size_t num_iters_no_change = 5,
        const double tol = 0.0001, 
        const bool shuffle = true, 
        const bool verbose = true,
        const bool multi_class = false): BaseOptimizer<DataType, 
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
                verbose, 
                multi_class) {};
    ~SGD() {};

    void optimize(const MatType& X, 
        const VecType& y) {
        
        std::size_t num_samples = X.rows(), num_features = X.cols();
        std::size_t num_batch = num_samples / this->batch_size_;
        std::size_t no_improvement_count = 0;

        bool is_converged = false;
        double best_loss = ConstType<double>::infinity();

        MatType X_new = X;
        VecType y_new = y;
        MatType grad;
        if (this->multi_class_) {
            std::set<DataType> label_set{y_new.begin(), y_new.end()};
            std::size_t num_classes = label_set.size();
            grad.resize(num_features, num_classes);
        }
        else {
            grad.resize(num_features, 1);
        }
        
        for (std::size_t iter = 0; iter < this->max_iter_; iter++) {
            if (this->shuffle_) {
                random::shuffle_data<MatType, VecType>(X_new, y_new, X_new, y_new);
            }
            MatType X_batch(this->batch_size_, num_features);
            VecType y_batch(this->batch_size_);
            VecType loss_history(num_batch);

            double lr = this->lr_decay_.compute(iter);

            for (std::size_t j = 0; j < num_batch; j++) {
                std::size_t begin = j * this->batch_size_;
                X_batch = X_new.middleRows(begin, this->batch_size_);
                y_batch = y_new.middleRows(begin, this->batch_size_);

                grad = this->loss_func_.gradient(X_batch, y_batch, this->x0_);
                // clip gradient with large value 
                grad = common::clip<MatType>(grad, this->MAX_DLOSS, this->MIN_DLOSS);
                // W = W - lr * grad; 
                this->x0_ = this->w_update_.update(this->x0_, grad, lr);
                double loss = this->loss_func_.evaluate(X_batch, y_batch, this->x0_);
                loss_history(j, 0) = loss;
            }
            // compute total loss from one batch
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
#endif /*CORE_OPTIMIZER_SGD_SGD_HPP*/
