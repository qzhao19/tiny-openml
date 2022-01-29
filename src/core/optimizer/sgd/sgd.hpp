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

    MatType X;
    VecType y;
    std::size_t max_iter;
    std::size_t batch_size;
    double alpha;
    double tol;
    bool shuffle;
    bool verbose;

    // internal callable parameters
    std::size_t num_samples;
    std::size_t num_features;
    std::size_t num_batch;

public:
    SGD(const MatType& X_, 
        const VecType& y_,
        const std::size_t max_iter_ = 20000, 
        const std::size_t batch_size_ = 16,
        const double alpha_ = 0.1,  
        const double tol_ = 0.0001, 
        const bool shuffle_ = false,
        const bool verbose_ = true): X(X_), y(y_), 
            max_iter(max_iter_), 
            batch_size(batch_size_), 
            tol(tol_), 
            alpha(alpha_), 
            shuffle(shuffle_), 
            verbose(verbose_) {
                num_samples = X_.rows();
                num_features = X_.cols();
                num_batch = num_samples / batch_size;
            };

    ~SGD() {};

    template<typename LossFuncionType, 
        typename UpdatePolicy>
    void optimize(LossFuncionType& loss_fn, 
        UpdatePolicy& update_policy,
        VecType& W) {
        
        double total_error = 0.0;
        VecType grad(num_features);
        
        for (std::size_t i = 0; i < max_iter; i++) {
            if (shuffle) {
                math::shuffle_data(X, y, X, y);
            }

            MatType X_batch(batch_size, num_features);
            VecType y_batch(batch_size);
            double error = 0.0;

            for (std::size_t j = 0; j < num_batch; j++) {
                std::size_t begin = j * batch_size;
                X_batch = X.middleRows(begin, batch_size);
                y_batch = y.middleRows(begin, batch_size);
                grad = loss_fn.gradient(X_batch, y_batch, W);
                // W = W - alpha * grad;
                update_policy.update(W, grad, W);
                error += loss_fn.evaluate(X_batch, y_batch, W);
            }

            double average_error = error / static_cast<double>(num_batch);
            if (std::abs(total_error - average_error) < tol) {
                break;
            } 
            total_error = average_error;
            if (verbose) {
                if ((i % 10) == 0) {
                    std::cout << "iter = " << i << ", loss value = " << average_error << std::endl;
                }
            }
        }
    }
};

}
}
#endif
