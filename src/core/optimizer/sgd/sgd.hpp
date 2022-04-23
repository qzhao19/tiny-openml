#ifndef CORE_OPTIMIZER_SGD_SGD_HPP
#define CORE_OPTIMIZER_SGD_SGD_HPP
#include "./update_policies/vanilla_update.hpp"
#include "./update_policies/momentum_update.hpp"
#include "../../../prereqs.hpp"
#include "../../../core.hpp"
using namespace openml;

namespace openml {
namespace optimizer {

template<typename DataType>
class SGD {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    MatType X_;
    VecType y_;
    std::size_t max_iter_;
    std::size_t batch_size_;
    double alpha_;
    double tol_;
    bool shuffle_;
    bool verbose_;

    // internal callable parameters
    std::size_t num_samples_;
    std::size_t num_features_;
    std::size_t num_batch_;

public:
    SGD(const MatType& X, 
        const VecType& y,
        const std::size_t max_iter = 20000, 
        const std::size_t batch_size = 16,
        const double alpha = 0.1,  
        const double tol = 0.0001, 
        const bool shuffle = true,
        const bool verbose = true): X_(X), y_(y), 
            max_iter_(max_iter), 
            batch_size_(batch_size), 
            tol_(tol), 
            alpha_(alpha), 
            shuffle_(shuffle), 
            verbose_(verbose) {
                num_samples_ = X.rows();
                num_features_ = X.cols();
                num_batch_ = num_samples_ / batch_size_;
            };

    ~SGD() {};

    template<typename LossFuncionType, 
        typename UpdatePolicy>
    void optimize(LossFuncionType& loss_fn, 
        UpdatePolicy& update_policy,
        VecType& W) {
        
        double total_error = 0.0;
        VecType grad(num_features_);
        
        for (std::size_t i = 0; i < max_iter_; i++) {
            if (shuffle_) {
                math::shuffle_data(X_, y_, X_, y_);
            }

            MatType X_batch(batch_size_, num_features_);
            VecType y_batch(batch_size_);
            double error = 0.0;

            for (std::size_t j = 0; j < num_batch_; j++) {
                std::size_t begin = j * batch_size_;
                X_batch = X.middleRows(begin, batch_size_);
                y_batch = y.middleRows(begin, batch_size_);

                grad = loss_fn.gradient(X_batch, y_batch, W);
                
                // W = W - alpha * grad;
                update_policy.update(W, grad, W);

                error += loss_fn.evaluate(X_batch, y_batch, W);
            }

            double average_error = error / static_cast<double>(num_batch_);
            if (std::abs(total_error - average_error) < tol_) {
                break;
            } 
            total_error = average_error;
            if (verbose_) {
                if ((i % 10) == 0) {
                    std::cout << "iter = " << i << ", loss value = " << average_error << std::endl;
                }
            }
        }
    }
};

}
}
#endif /*CORE_OPTIMIZER_SGD_SGD_HPP*/
