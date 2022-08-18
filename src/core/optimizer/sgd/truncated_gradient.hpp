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

template<typename DataType>
class TruncatedGradient {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    MatType X_;
    VecType y_;
    std::size_t max_iter_;
    std::size_t batch_size_;
    std::size_t num_iters_no_change_;
    DataType MAX_DLOSS = static_cast<DataType>(1e+10);
    
    double tol_;
    double alpha_;
    double l1_ratio_;
    bool shuffle_;
    bool verbose_;
    
    // internal callable parameters
    std::size_t num_samples_;
    std::size_t num_features_;
    std::size_t num_batch_;

    /**
     * truncated gradient implementation
    */
    void update(VecType& weight, 
        VecType& cum_l1,
        DataType max_cum_l1) {
        
        for (std::size_t j = 0; j < num_features_; ++j) {
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
    TruncatedGradient(const MatType& X, 
        const VecType& y,
        const std::size_t max_iter = 2000, 
        const std::size_t batch_size = 256,
        const std::size_t num_iters_no_change = 5,
        const double tol = 0.0001, 
        const double alpha = 0.0001,
        const double l1_ratio = 0.15,
        const bool shuffle = true, 
        const bool verbose = true): X_(X), y_(y), 
            max_iter_(max_iter), 
            batch_size_(batch_size), 
            num_iters_no_change_(num_iters_no_change),
            tol_(tol), 
            alpha_(alpha),
            l1_ratio_(l1_ratio),
            shuffle_(shuffle),
            verbose_(verbose) {
                num_samples_ = X_.rows();
                num_features_ = X_.cols();
                num_batch_ = num_samples_ / batch_size_;
            };
    ~TruncatedGradient() {};

     template<typename LossFuncionType, 
        typename UpdatePolicyType,
        typename DecayPolicyType>
    VecType optimize(const VecType& weights, 
        LossFuncionType& loss_fn, 
        UpdatePolicyType& w_update,
        DecayPolicyType& lr_decay) {
        
        bool is_converged = false;
        std::size_t no_improvement_count = 0;
        
        double best_loss = ConstType<double>::infinity();
        DataType max_cum_l1 = 0.0;
        VecType cum_l1(num_features_);
        cum_l1.setZero();

        VecType W = weights;
        VecType opt_W;
        for (std::size_t iter = 0; iter < max_iter_; iter++) {
            if (shuffle_) {
                random::shuffle_data<MatType, VecType>(X_, y_, X_, y_);
            }
            MatType X_batch(batch_size_, num_features_);
            VecType y_batch(batch_size_);
            VecType loss_history(batch_size_);

            double lr = lr_decay.compute(iter);

            for (std::size_t j = 0; j < num_batch_; j++) {
                std::size_t begin = j * batch_size_;
                X_batch = X_.middleRows(begin, batch_size_);
                y_batch = y_.middleRows(begin, batch_size_);

                VecType grad(num_features_);
                grad = loss_fn.gradient(X_batch, y_batch, W);

                // clip gradient with large value 
                grad = utils::clip<MatType>(grad, MAX_DLOSS, -MAX_DLOSS);

                // W = W - lr * grad; 
                W = w_update.update(W, grad, lr);

                max_cum_l1 += static_cast<DataType>(l1_ratio_) * 
                    static_cast<DataType>(lr) * static_cast<DataType>(alpha_);
                update(W, cum_l1, max_cum_l1);

                double loss = loss_fn.evaluate(X_batch, y_batch, W);

                loss_history(j, 0) = loss;
            }

            double sum_loss = static_cast<double>(loss_history.array().sum());

            if (sum_loss > best_loss - tol_ * batch_size_) {
                no_improvement_count +=1;
            }
            else {
                no_improvement_count = 0;
            }

            if (sum_loss < best_loss) {
                best_loss = sum_loss;
            }

            if (no_improvement_count >= num_iters_no_change_) {
                is_converged = true;
                opt_W = W;
                break;
            }

            if (verbose_) {
                if ((iter % 2) == 0) {
                    std::cout << "-- Epoch = " << iter << ", average loss value = " 
                            << sum_loss / static_cast<double>(batch_size_) << std::endl;
                }
            }
        }

        if (!is_converged) {
            std::ostringstream err_msg;
            err_msg << "Not converge, current number of epoch = " << max_iter_
                    << ", the batch size = " << batch_size_ 
                    << ", try apply different parameters." << std::endl;
            throw std::runtime_error(err_msg.str());
        }

        return opt_W;
    }
};

}
}
#endif /*CORE_OPTIMIZER_SGD_TRUNCATED_GRADIENT_HPP*/
