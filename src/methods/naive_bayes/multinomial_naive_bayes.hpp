#ifndef METHODS_NAIVE_BAYES_MULTINOMIAL_NAIVE_BAYES_HPP
#define METHODS_NAIVE_BAYES_MULTINOMIAL_NAIVE_BAYES_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "base.hpp"
using namespace openml;

namespace openml {
namespace naive_bayes {

template<typename DataType>
class MultinomialNB: public BaseNB<DataType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;
    double alpha_;

protected:
    std::size_t num_features_;
    MatType feature_count_;
    VecType class_count_;

    void compute_feature_prob(const MatType& X, 
        const VecType& y) {
        
        std::size_t num_samples = X.rows();
        num_features_ = X.cols();

        MatType feature_count(num_features_, this->num_classes_);
        VecType class_count(this->num_classes_);

        MatType X_y(num_samples, num_features_ + 1);
        X_y = common::hstack<MatType>(X, y);

        std::size_t new_i = 0;
        for (auto& label : this->label_map_) {
            MatType partial_X_y(label.second, num_features_ + 1);

            std::vector<std::size_t> keep_rows;
            for (std::size_t i = 0; i < X_y.rows(); ++i) {
                if (X_y(i, num_features_) == label.first) {
                    keep_rows.push_back(i);
                }
            }
            IdxVecType keep_cols = IdxVecType::LinSpaced(X_y.cols(), 0, X_y.cols());

            partial_X_y = X_y(keep_rows, keep_cols); 
            feature_count.col(new_i) = math::sum<MatType>(partial_X_y.leftCols(num_features_), 0);
            class_count.row(new_i) = math::sum<MatType>(partial_X_y.leftCols(num_features_));

            new_i++;
        }

        feature_count_ = feature_count.transpose();
        class_count_ = class_count.transpose();
    }


    /**
     * compute log likelihood function 
     * 
     * P(x|c) = (Ny_i + alpha) / (Ny + (alpha * num_features_))
     * 
     * α  is the smoothing parameter,
     * Ny_i  is the count of feature xi in class y
     * Ny  is the total count of all features in class y
    */
    const double compute_log_likelihood(const VecType& x, const std::size_t c) const {

        double retval = 0.0;
        for (std::size_t i = 0; i < x.rows(); ++i) {
            DataType numerator = feature_count_(c, i) + alpha_;
            DataType denominator = class_count_(c, 0) + (alpha_ * num_features_);
            retval += x(i) * std::log(numerator / denominator);
        }

        return static_cast<double>(retval);
    }

    const MatType joint_log_likelihood(const MatType& X) const {
        std::size_t num_samples = X.rows();
        // define log likelihood matrix
        MatType ll(num_samples, this->num_classes_);
        ll.setZero();

        VecType log_prior = this->prior_prob_.array().log();
        MatType lp = common::repeat<MatType>(log_prior.transpose(), num_samples, 0);
        for (std::size_t i = 0; i < num_samples; ++i) {        
            for (std::size_t c = 0; c < this->num_classes_; ++c) {
                VecType row = X.row(i).transpose();
                ll(i, c) = compute_log_likelihood(row, c);
            }
        }
        return ll + lp;
    };

public:
    /**
     * @param alpha float, default=1e-9
     *      Portion of the largest variance of all features.
    */
    MultinomialNB(): BaseNB<DataType>(), 
        alpha_(10.0) {};

    explicit MultinomialNB(double alpha) : BaseNB<DataType>(),
            alpha_(alpha) {};
    
    ~MultinomialNB() {};

    /**
     * Fit naive bayes classifier according to X, y.
     *  @param X ndarray of shape (num_samples, num_features_)
     *      input dataset
    */
    void fit(const MatType& X, 
        const VecType& y) {
        this->compute_prior_prob(y);
        compute_feature_prob(X, y);
    }

    /**
     * return uumber of samples encountered for each (class, feature) during fitting.
    */
    MatType get_features_count() {
        return feature_count_;
    }

};

} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_MULTINOMIAL_NAIVE_BAYES_HPP*/
