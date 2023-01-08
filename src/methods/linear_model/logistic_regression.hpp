#ifndef METHODS_LINEAR_MODEL_LOGISTIC_REGRESSION_HPP
#define METHODS_LINEAR_MODEL_LOGISTIC_REGRESSION_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
// #include "base.hpp"
using namespace openml;

namespace openml{
namespace linear_model {

/**
 * Logistic Regression classifier.
 * 
 * @param alpha  Learning rate when update weights
 * @param lambda L2 regularization coefficient, small values specify stronger 
 *               regularization.
 * @param tol    Tolerance for stopping criteria.
 * @param batch_size Number of samples that will be passed through the optimizer
 *                   specifies if we apply sgd method
 * @param max_iter Maximum number of iterations taken for the solvers to converge.
 * @param solver Algorithm to use in the optimization problem default "sgd"
 * @param penalty  penalty type {‘l1’, ‘l2’, ‘none’}, default=’l2’
 * @param update_policy sgd weight update policies
 * @param decay_policy weight decay type
*/
template<typename DataType>
class LogisticRegression {
private:
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    
    VecType w_;

    bool shuffle_;
    bool verbose_;
    double alpha_;
    double lambda_;
    double tol_;
    double l1_ratio_;
    double ftol_;
    double wolfe_;
    double delta_;
    std::size_t num_iters_no_change_;
    std::size_t max_linesearch_; 
    std::size_t batch_size_;
    std::size_t max_iter_;
    std::size_t mem_size_;
    std::size_t past_;
    std::string solver_;
    std::string penalty_;
    std::string update_policy_;
    std::string decay_policy_;
    std::string linesearch_policy_;
    std::string linesearch_condition_;

protected:
    /**
     * Train the logistic model with the given optimizer
     * logistic regression must have intercept term, because we want to 
     * find a decision boundary that is able to seperate 2 class data,
     * intercept term does not exist, the decision boundary no doubt 
     * pass through the origin point.
    */
    void fit_data(const MatType& X, 
        const VecType& y) {
        
        MatType X_new = X;
        VecType y_new = y;
        
        std::cout << "fit_data1" << std::endl;
        // std::size_t num_samples = X.rows(), num_features = X.cols();
        
        // VecType ones(num_samples);
        // ones.fill(1.0);
        // X_new.conservativeResize(num_samples, num_features + 1);
        // X_new.col(num_features) = ones;

        // VecType w(num_features + 1);
        // w.setOnes();

        std::size_t num_samples = X.rows();
        VecType ones(num_samples);
        ones.fill(1.0);
        X_new = utils::hstack<MatType>(X, ones);

        std::size_t num_features = X_new.cols();
        VecType w(num_features);
        w.setRandom();

        std::cout << "fit_data2" << std::endl;
        loss::LogLoss<DataType> log_loss;
        optimizer::VanillaUpdate<DataType> w_update;
        optimizer::StepDecay<DataType> lr_decay(0.1);
        
        std::cout << "fit_data3" << std::endl;
        optimizer::SGD<DataType, 
            loss::LogLoss<DataType>, 
            optimizer::VanillaUpdate<DataType>, 
            optimizer::StepDecay<DataType>> sgd(w, log_loss, w_update, lr_decay);
        
        std::cout << "fit_data4" << std::endl;
        sgd.optimize(X_new, y_new);
        std::cout << "fit_data5" << std::endl;
        w_ = sgd.get_coef();

        std::cout << w_ << std::endl;  
        // std::unique_ptr<optimizer::BaseOptimizer<
        //     DataType, 
        //     loss::LogLoss<DataType>, 
        //     optimizer::VanillaUpdate<DataType>, 
        //     optimizer::StepDecay<DataType>>> opt;     
        // opt = std::make_unique<optimizer::SGD<
        //     DataType, 
        //     loss::LogLoss<DataType>, 
        //     optimizer::VanillaUpdate<DataType>, 
        //     optimizer::StepDecay<DataType>>>(w, 
        //         log_loss, 
        //         w_update, 
        //         lr_decay, 
        //         max_iter_, 
        //         batch_size_, 
        //         num_iters_no_change_, 
        //         tol_, 
        //         shuffle_, 
        //         verbose_);
        // if (penalty_ == "l2" || penalty_ == "None") {
        //     if (solver_ == "sgd") {
        //         opt = std::make_unique<optimizer::SGD<
        //             DataType, 
        //             loss::LogLoss<DataType>, 
        //             optimizer::VanillaUpdate<DataType>, 
        //             optimizer::StepDecay<DataType>>>(w, 
        //                 log_loss, 
        //                 w_update, 
        //                 lr_decay, 
        //                 max_iter_, 
        //                 batch_size_, 
        //                 num_iters_no_change_, 
        //                 tol_, 
        //                 shuffle_, 
        //                 verbose_);
        //     }
        //     else if (solver_ == "sag") {
        //         opt = std::make_unique<optimizer::SAG<
        //             DataType, 
        //             loss::LogLoss<DataType>, 
        //             optimizer::VanillaUpdate<DataType>, 
        //             optimizer::StepDecay<DataType>>>(w, 
        //                 log_loss, 
        //                 w_update, 
        //                 lr_decay, 
        //                 max_iter_, 
        //                 batch_size_, 
        //                 num_iters_no_change_, 
        //                 tol_, 
        //                 shuffle_, 
        //                 verbose_);
        //     }
        //     else if (solver_ == "lbfgs") {
        //         optimizer::LineSearchParams<double> linesearch_params(
        //             ftol_, wolfe_, max_linesearch_, linesearch_condition_
        //         );
        //         opt = std::make_unique<optimizer::LBFGS<DataType, 
        //             loss::LogLoss<DataType>, 
        //             optimizer::LineSearchParams<DataType>>>(w, 
        //                 log_loss, 
        //                 linesearch_params,
        //                 max_iter_, 
        //                 mem_size_, 
        //                 past_,
        //                 tol_,
        //                 delta_,
        //                 linesearch_policy_,
        //                 shuffle_, 
        //                 verbose_);
        //     }
        //     else {
        //         throw std::invalid_argument("SGD, SAG and LBFGS solvers only support l2 regularization.");
        //     }
        // }
        // else if ("penality" == "l1") {
        //     if (solver_ == "scd") {
        //         opt = std::make_unique<optimizer::SCD<
        //             DataType, 
        //             loss::LogLoss<DataType>>>(w, 
        //                 log_loss, 
        //                 max_iter_, 
        //                 l1_ratio_,
        //                 lambda_, 
        //                 shuffle_,
        //                 verbose_);
        //     }
        //     else {
        //         throw std::invalid_argument("SCD solver only supports l1 regularization.");
        //     }
        // }
        // else {
        //     throw std::invalid_argument("Penalty type {l1, l2, none}, default=l2");
        // }
        
        // opt->optimize(X_new, y_new);
        // this->w_ = opt->get_coef();
    };

    /**Predict confidence scores for samples.*/
    // const VecType compute_decision_function(const MatType& X) const{

    //     std::size_t num_samples = X.rows(), num_features = X.cols();
    //     VecType decision_boundary(num_samples);

    //     decision_boundary = X * this->w_.topRows(num_features);
    //     VecType b(num_samples);
    //     b = utils::repeat<VecType>(this->w_.bottomRows(1), num_samples, 0);
    //     decision_boundary += b;

    //     return decision_boundary;
    // }

    // /** Predict class labels for samples in X.*/
    // const VecType predict_label(const MatType& X) const{
    
    //     // calculate the desicion boundary func
    //     std::size_t num_samples = X.rows();
    //     VecType decision_boundary(num_samples);
    //     VecType y_pred(num_samples);

    //     decision_boundary = compute_decision_function(X);
    //     y_pred = math::sigmoid(decision_boundary);
    //     for(auto& value:y_pred) {
    //         if (value > 0.5) {
    //             value = 0;
    //         }
    //         else {
    //             value = 1;
    //         }
    //     }
    //     return y_pred;
    // }

    // const MatType predict_label_prob(const MatType& X) const {
    //     // calculate the desicion boundary func
    //     std::size_t num_samples = X.rows();
    //     VecType decision_boundary(num_samples);
    //     decision_boundary = compute_decision_function(X);

    //     VecType ones(num_samples);
    //     ones.setOnes();

    //     VecType y_pred(num_samples);
    //     y_pred = math::sigmoid(decision_boundary);

    //     MatType prob(num_samples, 2);
    //     prob = utils::hstack<MatType>(y_pred, ones - y_pred);

    //     return prob;
    // }

public:
    // Constructor for sgd/sag
    // LogisticRegression(const double alpha, 
    //     const double lambda,
    //     const double tol, 
    //     const std::size_t batch_size, 
    //     const std::size_t max_iter, 
    //     const std::size_t num_iters_no_change,
    //     const std::string solver,
    //     const std::string penalty, 
    //     const std::string update_policy, 
    //     const std::string decay_policy, 
    //     bool shuffle = true, 
    //     bool verbose = true): BaseLinearModel<DataType>(true), 
    //         alpha_(alpha), 
    //         lambda_(lambda), 
    //         tol_(tol), 
    //         batch_size_(batch_size), 
    //         max_iter_(max_iter), 
    //         num_iters_no_change_(num_iters_no_change),
    //         solver_(solver),
    //         penalty_(penalty_), 
    //         update_policy_(update_policy), 
    //         decay_policy_(decay_policy), 
    //         shuffle_(shuffle), 
    //         verbose_(verbose) {};

    // Constructor for scd
    // LogisticRegression(const double lambda,
    //     const double l1_ratio,
    //     const std::size_t max_iter, 
    //     const std::string solver,
    //     const std::string penalty, 
    //     bool shuffle = true, 
    //     bool verbose = true): BaseLinearModel<DataType>(true), 
    //         lambda_(lambda), 
    //         l1_ratio_(l1_ratio),
    //         max_iter_(max_iter), 
    //         solver_(solver),
    //         penalty_(penalty_), 
    //         shuffle_(shuffle), 
    //         verbose_(verbose) {};    

    // Constructor for lbfgs
    // LogisticRegression(const double tol, 
    //     const double ftol,
    //     const double delta,
    //     const double wolfe,
    //     const std::size_t max_iter, 
    //     const std::size_t mem_size, 
    //     const std::size_t past,
    //     const std::size_t max_linesearch,
    //     const std::string solver,
    //     const std::string penalty, 
    //     const std::string linesearch_policy, 
    //     const std::string linesearch_condition, 
    //     bool shuffle = true, 
    //     bool verbose = true): BaseLinearModel<DataType>(true), 
    //         tol_(tol), 
    //         ftol_(ftol),
    //         delta_(delta),
    //         wolfe_(wolfe),
    //         max_iter_(max_iter), 
    //         mem_size_(mem_size),
    //         past_(past),
    //         max_linesearch_(max_linesearch),
    //         solver_(solver),
    //         penalty_(penalty_), 
    //         linesearch_policy_(linesearch_policy), 
    //         linesearch_condition_(linesearch_condition), 
    //         shuffle_(shuffle), 
    //         verbose_(verbose) {};

    // Default constructor
    LogisticRegression(): alpha_(0.1), 
        lambda_(0.0), 
        tol_(0.0001), 
        batch_size_(16), 
        max_iter_(2000), 
        num_iters_no_change_(5),
        solver_("sgd"),
        penalty_("None"), 
        update_policy_("vanilla"), 
        decay_policy_("constant"),
        shuffle_(true), 
        verbose_(true) {};

    ~LogisticRegression() {};


    void fit(const MatType& X, 
        const VecType& y) {
        fit_data(X, y);
    }


    
    /**
     * Probability estimates. The returned estimates for all 
     * classes are ordered by the label of classes.
     * 
     * @param X ndarray of shape [num_samples, num_features], 
     *      The data matrix for which we want to get the predictions.
     * @return Returns the probability of the sample for each class in the model
    */
    // const MatType predict_prob(const MatType& X) const {
    //     std::size_t num_samples = X.rows();
    //     MatType prob(num_samples, 2);
    //     prob = predict_label_prob(X);
    //     return prob;
    // }
};

} // namespace openml
} // namespace regression

#endif /*METHODS_LINEAR_MODEL_LOGISTIC_REGRESSION_HPP*/
