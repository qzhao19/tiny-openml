#ifndef METHODS_NAIVE_BAYES_GAUSSIAN_NAIVE_BAYES_HPP
#define METHODS_NAIVE_BAYES_GAUSSIAN_NAIVE_BAYES_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "base.hpp"
using namespace openml;

namespace openml {
namespace naive_bayes {

template<typename DataType>
class GaussianNB: public BaseNB<DataType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    double var_smoothing_;

protected:
    /**
     * @param means Matrix of shape [n_classes, n_feature], 
     *      measn of each feature for different class
     * @param vars  Matrix of shape [n_classes, n_feature], 
     *      variances of each feature for different class
    */
    MatType mean_;
    MatType var_;


    /**
     * compute log posterior prbabilty, P(c|x) = P(c)P(x|c)
     * log(P(c|x)) = log(P(C)) + log(P(x|c)) for all rows x of X, 
     * as an array-like of shape (n_classes, n_samples).
    */
    const MatType joint_log_likelihood(const MatType& X) const {
        std::size_t num_samples = X.rows();
        MatType jll(num_samples, this->num_classes_);
        jll.setZero();

        VecType log_prior = this->prior_prob_.array().log();
        MatType lp = common::repeat<MatType>(log_prior.transpose(), num_samples, 0);
        
        for (std::size_t i = 0; i < this->num_classes_; i++) {
            auto val = var_.row(i) * 2.0 * M_PI;

            DataType sum_log_var = -0.5 * val.array().log().sum();
            MatType X_centered = (X.rowwise() - mean_.row(i)).array().square();
            MatType X_centered_reduced = X_centered.array().rowwise() / var_.row(i).array();
            VecType ll = sum_log_var - X_centered_reduced.array().rowwise().sum() * 0.5;

            jll.col(i) = ll.array() + lp.array();
        }

        return jll;
    };
    
    /**
     * Compute Gaussian mean and variance
    */
    void update_mean_variance(const MatType& X, 
        const VecType& y) {
        std::size_t num_samples = X.rows(), num_features = X.cols();

        MatType mean(num_features, this->num_classes_);
        MatType var(num_features, this->num_classes_);

        MatType X_y(num_samples, num_features + 1);
        X_y = common::hstack<MatType>(X, y);
        
        std::size_t new_i = 0;
        for (auto& label : this->label_map_) {
            MatType partial_X_y(label.second, num_features + 1);
            
            std::vector<std::size_t> keep_rows;
            for (std::size_t i = 0; i < X_y.rows(); ++i) {
                if (X_y(i, num_features) == label.first) {
                    keep_rows.push_back(i);
                }
            }
            IdxType keep_cols = IdxType::LinSpaced(X_y.cols(), 0, X_y.cols());

            partial_X_y = X_y(keep_rows, keep_cols); 
            mean.col(new_i) = math::mean<MatType, VecType>(partial_X_y.leftCols(num_features), 0);
            var.col(new_i) = math::var<MatType, VecType>(partial_X_y.leftCols(num_features), 0);

            new_i++;
        }
        mean_ = mean.transpose();
        var_ = var.transpose();
    }

public:
    /**
     * @param var_smoothing float, default=1e-9
     *      Portion of the largest variance of all features.
    */
     GaussianNB(): BaseNB<DataType>(), 
        var_smoothing_(1e-9){};

    explicit GaussianNB(double var_smoothing) : BaseNB<DataType>(),
            var_smoothing_(var_smoothing){};
    
    ~GaussianNB() {};

    /**
     * Fit the model with X.
     * @param X array-like of shape (num_samples, num_features)
     *      Training data, where num_samples is the number of 
     *      samples and num_features is the number of features.
    */
    void fit(const MatType& X, 
        const VecType& y) {
        
        this->compute_prior_prob(y);
        this->update_mean_variance(X, y);
        var_ = var_.array() + var_smoothing_ * var_.maxCoeff();
    };

    /** get the mean attributes */
    const VecType get_mean() const {
        return mean_;
    }

    /** get the var attributes */
    const VecType get_var() const {
        return var_;
    }

};

} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_GAUSSIAN_NAIVE_BAYES_HPP*/
