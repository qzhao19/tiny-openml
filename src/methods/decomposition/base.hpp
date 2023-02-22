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
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

protected:
    MatType precision_;
    MatType covariance_;
    MatType components_;
    VecType explained_var_;
    VecType explained_var_ratio_;
    std::size_t num_components_;


public:
    BaseDecompositionModel(): num_components_(2) {};
    BaseDecompositionModel(const std::size_t num_components): 
        num_components_(num_components) {};

    ~BaseDecompositionModel() {};

    /**override get_coef interface*/
    const VecType get_components() const {
        return components_;
    };

    const VecType get_explained_var() const {
        return explained_var_;
    };

    const VecType get_explained_var_ratio() const {
        return explained_var_ratio_;
    };

};

} // namespace openml
} // namespace regression

#endif /*METHODS_DECOMPOSITION_BASE_HPP*/
