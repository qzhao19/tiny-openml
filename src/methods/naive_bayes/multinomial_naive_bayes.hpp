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


    }


public:
    

};

} // naive_bayes
} // openml

#endif /*METHODS_NAIVE_BAYES_GAUSSIAN_NAIVE_BAYES_HPP*/
