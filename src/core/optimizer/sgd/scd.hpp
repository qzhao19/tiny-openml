#ifndef CORE_OPTIMIZER_SGD_SCD_HPP
#define CORE_OPTIMIZER_SGD_SCD_HPP
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
template<typename DataType, 
    typename LossFuncionType, 
    typename UpdatePolicyType = optimizer::VanillaUpdate<DataType>,
    typename DecayPolicyType = optimizer::StepDecay<DataType>>
class SCD: public BaseOptimizer<DataType, 
    LossFuncionType, 
    UpdatePolicyType, 
    DecayPolicyType> {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;

    double lambda_;
    double l1_ratio_;
    
public:
    SCD(const VecType& x0,
        const LossFuncionType& loss_func,
        const std::size_t max_iter = 5000, 
        const double l1_ratio = 1.0, 
        const double lambda = 0.0001,
        const bool shuffle = true,
        const bool verbose = true): BaseOptimizer<DataType, 
            LossFuncionType, 
            UpdatePolicyType, 
            DecayPolicyType>(x0, 
                loss_func, 
                max_iter, 
                shuffle, 
                verbose),
            lambda_(lambda),
            l1_ratio_(l1_ratio) {};
    ~SCD() {};


    void optimize(const MatType& X, 
        const VecType& y) {
        
        double eta = 0.0;
        std::size_t feat_index = 0;
        std::size_t num_samples = X.rows(), num_features = X.cols();

        MatType X_new = X;
        VecType y_new = y;
        VecType grad(num_features);

        for (std::size_t iter = 0; iter < this->max_iter_; iter++) {
            if (this->shuffle_) {
                random::shuffle_data<MatType, VecType>(X_new, y_new, X_new, y_new);
            }

            grad = this->loss_func_.gradient(X_new, y_new, this->x0_);

            double pred_descent = 0.0;
            double best_descent = -1.0;
            double best_eta = 0.0;
            std::size_t best_index = 0.0;
            
            for (feat_index = 0; feat_index < num_features; ++feat_index) {
                if ((this->x0_(feat_index, 0) - grad(feat_index, 0) / l1_ratio_) > (lambda_ / l1_ratio_)) {
                    eta = (-grad(feat_index, 0) / l1_ratio_) - (lambda_ / l1_ratio_);
                }
                else if ((this->x0_(feat_index, 0) - grad(feat_index, 0) / l1_ratio_) < (-lambda_ / l1_ratio_)) {
                    eta = (-grad(feat_index, 0) / l1_ratio_) + (lambda_ / l1_ratio_);
                }
                else {
                    eta = -this->x0_(feat_index, 0);
                }

                pred_descent = -eta * grad(feat_index, 0) - 
                    l1_ratio_ / 2 * eta * eta - 
                        lambda_ * std::abs(this->x0_(feat_index, 0) + eta) + 
                            lambda_ * std::abs(this->x0_(feat_index, 0));

                if (pred_descent > best_descent) {
                    best_descent = pred_descent;
                    best_index = feat_index;
                    best_eta = eta;
                }
            }
            feat_index = best_index;
            eta = best_eta;
            this->x0_(feat_index, 0) += eta;

            if (this->verbose_) {
                if ((iter % 100) == 0) {
                    double w_norm = this->x0_.array().abs().sum();
                    double loss = this->loss_func_.evaluate(X_new, y_new, this->x0_);
                    std::cout << "-- Epoch = " << iter << ", weight norm = " 
                        << w_norm <<", loss value = " << loss / num_samples << std::endl;
                }
            }
        }
        this->opt_x_ = this->x0_;
    } 
};

}
}
#endif /*CORE_OPTIMIZER_SGD_SCD_HPP*/
