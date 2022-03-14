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

    MatType feature_count;
    VecType class_count;

    void compute_feature_prob(const MatType& X, 
        const VecType& y) {
        
        std::size_t num_samples = X.rows(), num_features = X.cols();

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





    const double log_likelihood(const VecType& x, const std::size_t c) {

        double retval = 0.0;
        for (std::size_t i = 0; i < x.rows(); i++) {
            DataType numerator = feature_count(c, i)
        }


    }


    const MatType joint_log_likelihood(const MatType& X) const {
        std::size_t num_samples = X.rows();
        MatType jll(num_samples, this->num_classes);
        jll.setZero();

        for (std::size_t i = 0; i < this->num_classes; i++) {
            
            VecType log_prior = this->prior_prob.array().log();
            MatType log_prior_mat = utils::repeat<MatType>(log_prior.transpose(), num_samples, 0);

            



        }

        return jll;
    };

public:
    /**
     * @param alpha float, default=1e-9
     *      Portion of the largest variance of all features.
    */
    MultinomialNB(): BaseNB<DataType>(), 
        alpha(1.0){};

    MultinomialNB(const double alpha_) : BaseNB<DataType>(),
            alpha(alpha_){};
    
    ~MultinomialNB() {};


    void test_func(const MatType& X, 
        const VecType& y) {
        
        this->compute_prior_prob(y);

        compute_feature_prob(X, y);

    }

};

} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_MULTINOMIAL_NAIVE_BAYES_HPP*/
