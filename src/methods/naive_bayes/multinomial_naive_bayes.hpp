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
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    double alpha;

protected:
    std::size_t num_features;
    MatType feature_count;
    VecType class_count;

    void compute_feature_prob(const MatType& X, 
        const VecType& y) {
        
        std::size_t num_samples = X.rows();
        num_features = X.cols();

        MatType feature_count_(num_features, this->num_classes);
        VecType class_count_(this->num_classes);

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
            feature_count_.col(new_i) = math::sum<MatType, VecType>(partial_X_y.leftCols(num_features), 0);
            class_count_.row(new_i) = math::sum<MatType, VecType>(partial_X_y.leftCols(num_features));

            new_i++;
        }

        feature_count = feature_count_.transpose();
        class_count = class_count_.transpose();
    }


    /**
     * compute log likelihood function 
     * 
     * P(x|c) = (Ny_i + alpha) / (Ny + (alpha * num_features))
     * 
     * α  is the smoothing parameter,
     * Ny_i  is the count of feature xi in class y
     * Ny  is the total count of all features in class y
    */
    const double compute_log_likelihood(const VecType& x, const std::size_t c) const {

        double retval = 0.0;
        for (std::size_t i = 0; i < x.rows(); ++i) {
            DataType numerator = feature_count(c, i) + alpha;
            DataType denominator = class_count(c, 0) + (alpha * num_features);
            retval += x(i) * std::log(numerator / denominator);
        }

        return static_cast<double>(retval);
    }

    const MatType joint_log_likelihood(const MatType& X) const {
        std::size_t num_samples = X.rows();
        // define log likelihood matrix
        MatType ll(num_samples, this->num_classes);
        ll.setZero();

        VecType log_prior = this->prior_prob.array().log();
        MatType lp = utils::repeat<MatType>(log_prior.transpose(), num_samples, 0);
        for (std::size_t i = 0; i < num_samples; ++i) {        
            for (std::size_t c = 0; c < this->num_classes; ++c) {
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
        alpha(10.0) {};

    explicit MultinomialNB(const double alpha_) : BaseNB<DataType>(),
            alpha(alpha_) {};
    
    ~MultinomialNB() {};

    /**
     * Fit naive bayes classifier according to X, y.
     *  @param X ndarray of shape (num_samples, num_features)
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
        return feature_count;
    }

};

} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_MULTINOMIAL_NAIVE_BAYES_HPP*/
