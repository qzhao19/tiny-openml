#ifndef CORE_OPTIMIZER_SAG_SAG_HPP
#define CORE_OPTIMIZER_SAG_SAG_HPP
#include "../sgd/decay_policies/step_decay.hpp"
#include "../sgd/decay_policies/exponential_decay.hpp"
#include "../sgd/update_policies/vanilla_update.hpp"
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
    
    double tol_;
    bool shuffle_;
    bool verbose_;
    
    // internal callable parameters
    std::size_t num_samples_;
    std::size_t num_features_;
    std::size_t num_batch_;

public:
    SAG(const MatType& X, 
        const VecType& y,
        const std::size_t max_iter = 2000, 
        const std::size_t batch_size = 16,
        const double tol = 0.001, 
        const bool shuffle = true, 
        const bool verbose = true): X_(X), y_(y), 
            max_iter_(max_iter), 
            batch_size_(batch_size), 
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
        
        bool is_converged = false;
        double best_loss = 0.0;

        // define a matirx to store gradient history and a valiable of average gradient
        MatType grad_history(num_features_, num_batch_);
        grad_history.setZero();
        VecType avg_grad = math::mean<MatType, VecType>(grad_history, 1);

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

                // update average gradient, then replace with new grad
                VecType tmp = grad - grad_history.col(j);
                avg_grad.array() += tmp.array() / static_cast<DataType>(batch_size_);
                grad_history.col(j) = grad;

                // W = W - lr * grad; 
                W = w_update.update(W, avg_grad, lr);
                double loss = loss_fn.evaluate(X_batch, y_batch, W);

                loss_history(j, 0) = loss;
            }

            double avg_loss = static_cast<double>(loss_history.array().mean());
            
            if (std::abs(best_loss - avg_loss) < tol_) {
                is_converged = true;
                opt_W = W;
                break;
            } 

            best_loss = avg_loss;
            if (verbose_) {
                if ((iter % 10) == 0) {
                    std::cout << "-- Epoch = " << iter << ", loss value = " << best_loss << std::endl;
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
#endif /*CORE_OPTIMIZER_SAG_SAG_HPP*/
