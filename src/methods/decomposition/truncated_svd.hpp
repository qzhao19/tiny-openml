#ifndef METHODS_DECOMPOSITION_TRUNCATED_SVD_HPP
#define METHODS_DECOMPOSITION_TRUNCATED_SVD_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
#include "./base.hpp"

using namespace openml;

namespace openml{
namespace decomposition {

/**
 * linear dimesionality reduction using randomized SVD of the data to project it 
 * to a lower dimesional space, input data shoule be centered 
 *  
 * @param n_components Number of components to keep
 * @param 
*/
template<typename DataType>
class TruncatedSVD: public BaseDecompositionModel<DataType> {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    std::size_t num_iters_;
    std::size_t num_oversamples_;
    std::string solver_;
    std::string power_iter_normalizer_;
    
protected:
    void fit_data(const MatType& X) {       
        VecType s;
        MatType U, Vt, X_transformed ; 
        if (solver_ == "randomized") {
            std::tie(U, s, Vt) = math::randomized_svd<MatType, VecType, IdxType>(X, 
                this->num_components_, 
                num_oversamples_, 
                num_iters_, 
                power_iter_normalizer_, 
                true);
            X_transformed = X * Vt.transpose();
        }

        // compute variance of each faeture along axis = 0
        double full_var;
        full_var = math::var<MatType>(X, 0).sum();

        this->components_ = Vt;
        this->singular_values_ = s;
        this->explained_var_ = math::var<MatType>(X_transformed, 0).transpose();
        this->explained_var_ratio_ = this->explained_var_.array() / full_var;
    }

    /** transform the new data */
    const MatType transform_data(const MatType& X) const{
        std::size_t num_samples = X.rows();
        
        MatType transformed_X(num_samples, this->num_components_);
        transformed_X = X * this->components_.transpose();
        return transformed_X; 
    }

public:
    TruncatedSVD(): BaseDecompositionModel<DataType>(2),
        num_oversamples_(2), 
        num_iters_(4),
        solver_("randomized"), 
        power_iter_normalizer_("LU") {};

    TruncatedSVD(std::size_t num_components, 
        std::size_t num_oversamples,
        std::size_t num_iters,
        std::string solver, 
        std::string power_iter_normalizer): BaseDecompositionModel<DataType>(num_components),
            num_oversamples_(num_oversamples), 
            num_iters_(num_iters),
            solver_(solver), 
            power_iter_normalizer_(power_iter_normalizer) {};

};  

} // namespace openml
} // namespace decomposition

#endif /*METHODS_DECOMPOSITION_TRUNCATED_SVD_HPP*/
