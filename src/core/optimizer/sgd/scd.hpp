#ifndef CORE_OPTIMIZER_SGD_COORDINATE_DESCENT_HPP
#define CORE_OPTIMIZER_SGD_COORDINATE_DESCENT_HPP
#include "./decay_policies/step_decay.hpp"
#include "./decay_policies/exponential_decay.hpp"
#include "./update_policies/vanilla_update.hpp"
#include "./update_policies/momentum_update.hpp"
#include "../../../prereqs.hpp"
#include "../../../core.hpp"
using namespace openml;

namespace openml {
namespace optimizer {

/**
 * Stochastic coordinate descent algorithm
*/
template<typename DataType>
class SCD {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    MatType X_;
    VecType y_;
    std::size_t max_iter_;
    
    double rho_;
    double alpha_;
    
    bool shuffle_;
    bool verbose_;

    std::size_t num_samples_;
    std::size_t num_features_;

public:
    SCD(const MatType& X, 
        const VecType& y,
        const std::size_t max_iter = 50, 
        const double rho = 1.0, 
        const double alpha = 0.001, 
        const bool shuffle = false,
        const bool verbose = true): X_(X), y_(y), 
            max_iter_(max_iter), 
            rho_(rho), 
            alpha_(alpha),
            shuffle_(shuffle), 
            verbose_(verbose) {
                num_samples_ = X_.rows();
                num_features_ = X_.cols();
            };
    ~SCD() {};

    template<typename LossFuncionType>
    VecType optimize(const VecType& weights, 
        LossFuncionType& loss_fn) {
        
        double eta = 0;
        std::size_t feature_index = 0;
        
        VecType grad(num_features_);
        VecType W = weights;

        for (std::size_t iter = 0; iter < max_iter_; iter++) {
            if (shuffle_) {
                random::shuffle_data<MatType, VecType>(X_, y_, X_, y_);
            }

            grad = loss_fn.gradient(X_, y_, W);

            std::cout << grad.transpose() << std::endl;

            double pred_descent = 0.0;
            double best_descent = -1.0;
            double best_eta = 0.0;
            std::size_t best_index = 0.0;
            
            for (feature_index = 0; feature_index < num_features_; ++feature_index) {
                if ((W(feature_index, 0) - grad(feature_index, 0) / rho_) > (alpha_ / rho_)) {
                    eta = (-grad(feature_index, 0) / rho_) - (alpha_ / rho_);
                }
                else if ((W(feature_index, 0) - grad(feature_index, 0) / rho_) < (-alpha_ / rho_)) {
                    eta = (-grad(feature_index, 0) / rho_) + (alpha_ / rho_);
                }
                else {
                    eta = -W(feature_index, 0);
                }

                pred_descent = -eta * grad(feature_index, 0) - 
                    rho_ / 2 * eta * eta - 
                        alpha_ * std::abs(W(feature_index, 0) + eta) + 
                            alpha_ * std::abs(W(feature_index, 0));

                if (pred_descent > best_descent) {
                    best_descent = pred_descent;
                    best_index = feature_index;
                    best_eta = eta;
                }
            }
            feature_index = best_index;
            eta = best_eta;
            W(feature_index, 0) += eta;
        }
        return W;
    } 
};

}
}
#endif /*CORE_OPTIMIZER_SGD_COORDINATE_DESCENT_HPP*/
