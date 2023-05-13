#ifndef METHODS_DECOMPOSITION_BASE_HPP
#define METHODS_DECOMPOSITION_BASE_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml{
namespace decomposition {

template<typename DataType>
class BaseDecompositionModel {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxVecType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

protected:
    MatType precision_;
    MatType covariance_;
    MatType components_;
    VecType explained_var_;
    VecType singular_values_;
    VecType explained_var_ratio_;
    std::size_t num_components_;

    virtual void fit_data(const MatType& X) = 0;

    virtual const MatType transform_data(const MatType& X) const = 0;

public:
    BaseDecompositionModel(): num_components_(2) {};
    BaseDecompositionModel(const std::size_t num_components): 
        num_components_(num_components) {};

    ~BaseDecompositionModel() {};

    /**
     * Fit the model with X.
     * @param X array-like of shape (num_samples, num_features)
     *      Training data, where num_samples is the number of 
     *      samples and num_features is the number of features.
    */
    void fit(const MatType& X) {
        this->fit_data(X);
    }

    /**
     * Apply dimensionality reduction to X. X is projected on the 
     * first principal components previously extracted from a 
     * training set.
     * 
     * @param X array-like of shape (num_samples, num_features)
     *      New data
     * @return X_new array-like of shape
     *      Projection of X in the first principal components
    */
    const MatType transform(const MatType& X) const{
        MatType transformed_X;
        transformed_X = this->transform_data(X); 
        return transformed_X;
    }

    /**
     * The right singular vectors of the input data.
    */
    const MatType get_components() const {
        return components_;
    };

    /**
     * The variance of the training samples transformed by a projection to each component
    */
    const VecType get_explained_var() const {
        return explained_var_;
    };

    /**
     * Percentage of variance explained by each of the selected components.
    */
    const VecType get_explained_var_ratio() const {
        return explained_var_ratio_;
    };

    /**
     * The singular values corresponding to each of the selected components.
    */
    const VecType get_singular_values() const {
        return singular_values_;
    };

};

} // namespace openml
} // namespace regression

#endif /*METHODS_DECOMPOSITION_BASE_HPP*/
