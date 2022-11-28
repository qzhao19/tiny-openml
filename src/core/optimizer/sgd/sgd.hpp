#ifndef CORE_OPTIMIZER_SGD_SGD_HPP
#define CORE_OPTIMIZER_SGD_SGD_HPP
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
class SGD {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    VecType x0_;
    std::size_t max_iter_;
    std::size_t batch_size_;
    std::size_t num_iters_no_change_;
    double tol_;
    bool shuffle_;
    bool verbose_;
    
    // internal callable parameters
    DataType MAX_DLOSS = static_cast<DataType>(1e+10);

public:
    SGD(
        const std::size_t max_iter = 2000,
        const std::size_t batch_size = 16,
        const std::size_t num_iters_no_change = 5,
        const double tol = 0.0001, 
        const bool shuffle = true, 
        const bool verbose = true):
            max_iter_(max_iter), 
            batch_size_(batch_size),
            num_iters_no_change_(num_iters_no_change),
            tol_(tol), 
            shuffle_(shuffle),
            verbose_(verbose) {};
    ~SGD() {};

    VecType optimize(const MatType& X, 
        const VecType& y, 
        const VecType& weight) {
        
        LossFuncionType loss_fn;
        UpdatePolicyType w_update;
        DecayPolicyType lr_decay;

        std::size_t num_samples = X.rows(), num_features = X.cols();
        std::size_t num_batch = num_samples / batch_size_;
        std::size_t no_improvement_count = 0;

        bool is_converged = false;
        double best_loss = ConstType<double>::infinity();

        MatType X_new = X;
        VecType y_new = y;
        VecType grad(num_features);
        VecType w = weight;
        VecType opt_w;
        for (std::size_t iter = 0; iter < max_iter_; iter++) {
            if (shuffle_) {
                random::shuffle_data<MatType, VecType>(X_new, y_new, X_new, y_new);
            }
            MatType X_batch(batch_size_, num_features);
            VecType y_batch(batch_size_);
            VecType loss_history(batch_size_);

            double lr = lr_decay.compute(iter);

            for (std::size_t j = 0; j < num_batch; j++) {
                std::size_t begin = j * batch_size_;
                X_batch = X_new.middleRows(begin, batch_size_);
                y_batch = y_new.middleRows(begin, batch_size_);

                grad = loss_fn.gradient(X_batch, y_batch, w);
                // clip gradient with large value 
                grad = utils::clip<MatType>(grad, MAX_DLOSS, -MAX_DLOSS);

                // W = W - lr * grad; 
                w = w_update.update(w, grad, lr);
                double loss = loss_fn.evaluate(X_batch, y_batch, w);
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
                opt_w = w;
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

        return opt_w;
    }

};

}
}
#endif
