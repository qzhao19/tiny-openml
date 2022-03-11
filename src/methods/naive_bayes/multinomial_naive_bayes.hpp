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


    MatType feature_count;
    VecType class_count;


protected:
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
            class_count_.col(new_i) = math::sum<MatType, VecType>(partial_X_y.leftCols(num_features));
        }

        std::cout << feature_count_ << std::endl;
        std::cout << class_count_ << std::endl;
    }


public:
    

};

} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_MULTINOMIAL_NAIVE_BAYES_HPP*/
