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

    double var_smoothing;

    /**
     * @param means Matrix of shape [n_classes, n_feature], 
     *      measn of each feature for different class
     * @param vars  Matrix of shape [n_classes, n_feature], 
     *      variances of each feature for different class
    */
    MatType mean;
    MatType var;

protected:

    /**
     * compute log posterior prbabilty, P(c|x) = P(c)P(x|c)
     * log(P(c|x)) = log(P(C)) + log(P(x|c)) for all rows x of X, 
     * as an array-like of shape (n_classes, n_samples).
    */
    const MatType joint_log_likelihood(const MatType& X) const {
        std::size_t num_samples = X.rows();
        MatType jll(num_samples, this->num_classes);
        jll.setZero();

        for (std::size_t i = 0; i < this->num_classes; i++) {
            
            VecType log_prior = this->prior_prob.array().log();
            MatType jointi = utils::repeat<MatType>(log_prior.transpose(), num_samples, 0);

            auto val1 = var.row(i) * 2.0 * M_PI;

            VecType sum_log_var = math::sum<MatType, VecType>(val1.array().log().matrix()) * (-0.5);
            MatType repeated_sum_log_var = utils::repeat<MatType>(sum_log_var, num_samples, 0);
            
            MatType X_minus_mean = X.array() - utils::repeat<MatType>(mean.row(i), num_samples, 0).array();
            MatType X_minus_mean_squared = X_minus_mean.array().square();
            MatType repeated_var = utils::repeat<MatType>(var.row(i), num_samples, 0);

            MatType X_minus_mean_div_var = X_minus_mean_squared.array() / repeated_var.array();
            VecType n_ij = repeated_sum_log_var.array() - 0.5 * math::sum<MatType, VecType>(X_minus_mean_div_var, 1).array();

            jll.col(i) = n_ij.array() + jointi.array();
            
        }

        return jll;
    };
    
    /**
     * Compute Gaussian mean and variance
    */
    void update_mean_variance(const MatType& X, 
        const VecType& y) {
        std::size_t num_samples = X.rows(), num_features = X.cols();

        MatType mean_(num_features, this->num_classes);
        // mean_.setZero();
        MatType var_(num_features, this->num_classes);
        // var_.setZero();

        MatType X_y(num_samples, num_features + 1);
        X_y = utils::hstack<MatType>(X, y);
        
        std::size_t new_i = 0;
        for (auto& label : this->label_map) {
            MatType partial_X_y(label.second, num_features + 1);
            
            std::vector<std::size_t> keep_rows;
            for (std::size_t i = 0; i < X_y.rows(); ++i) {
                if (X_y(i, num_features) == label.first) {
                    keep_rows.push_back(i);
                }
            }
            IdxType keep_cols = IdxType::LinSpaced(X_y.cols(), 0, X_y.cols());

            partial_X_y = X_y(keep_rows, keep_cols); 
            mean_.col(new_i) = math::mean<MatType, VecType>(partial_X_y.leftCols(num_features), 0);
            var_.col(new_i) = math::var<MatType, VecType>(partial_X_y.leftCols(num_features), 0);

            new_i++;
        }
        mean = mean_.transpose();
        var = var_.transpose();
    }



public:
    /**
     * @param var_smoothing float, default=1e-9
     *      Portion of the largest variance of all features.
    */
     GaussianNB(): BaseNB<DataType>(), 
        var_smoothing(1e-9){};

    GaussianNB(const double var_smoothing_) : BaseNB<DataType>(),
            var_smoothing(var_smoothing_){};
    
    ~GaussianNB() {};

    void fit(const MatType& X, 
        const VecType& y) {
        
        this->get_prior_prob(y);
        this->update_mean_variance(X, y);
        var = var.array() + var_smoothing * var.maxCoeff();
    };

};

} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_GAUSSIAN_NAIVE_BAYES_HPP*/
