#ifndef CORE_OPTIMIZER_SGD_SAG_HPP
#define CORE_OPTIMIZER_SGD_SAG_HPP
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
class SAG {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    MatType X_;
    VecType y_;
    std::size_t max_iter_;
    std::size_t batch_size_;
    std::size_t num_iters_no_change_;
    
    double tol_;
    bool shuffle_;
    bool verbose_;
    
    // internal callable parameters
    std::size_t num_samples_;
    std::size_t num_features_;
    std::size_t num_batch_;
    DataType MAX_DLOSS = static_cast<DataType>(1e+10);

public:
    SAG(const MatType& X, 
        const VecType& y,
        const std::size_t max_iter = 2000, 
        const std::size_t batch_size = 300,
        const std::size_t num_iters_no_change = 5,
        const double tol = 0.0001, 
        const bool shuffle = true, 
        const bool verbose = true): X_(X), y_(y), 
            max_iter_(max_iter), 
            batch_size_(batch_size), 
            num_iters_no_change_(num_iters_no_change),
            tol_(tol), 
            shuffle_(shuffle),
            verbose_(verbose) {
                num_samples_ = X_.rows();
                num_features_ = X_.cols();
                num_batch_ = num_samples_ / batch_size_;
            };
    ~SAG() {};

    template<typename LossFuncionType, 
        typename UpdatePolicyType,
        typename DecayPolicyType>
    VecType optimize(const VecType& weights, 
        LossFuncionType& loss_fn, 
        UpdatePolicyType& w_update,
        DecayPolicyType& lr_decay) {
        
        std::size_t no_improvement_count = 0;
        bool is_converged = false;
        double best_loss = ConstType<double>::infinity();

        // define a matirx to store gradient history and a valiable of average gradient
        MatType grad_history(num_features_, num_batch_);
        grad_history.setZero();
        VecType avg_grad = math::mean<MatType, VecType>(grad_history, 1);
        VecType grad(num_features_);

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

                grad = loss_fn.gradient(X_batch, y_batch, W);
                grad = utils::clip<MatType>(grad, MAX_DLOSS, -MAX_DLOSS);

                // update average gradient, then replace with new grad
                // VecType tmp = grad - grad_history.col(j);
                avg_grad = avg_grad + ((grad - grad_history.col(j)) / static_cast<DataType>(batch_size_));
                grad_history.col(j) = grad;

                // W = W - lr * grad; 
                W = w_update.update(W, avg_grad, lr);
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
#endif /*CORE_OPTIMIZER_SGD_SAG_HPP*/
