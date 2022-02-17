#ifndef METHODS_DECOMPOSITION_PCA_HPP
#define METHODS_DECOMPOSITION_PCA_HPP
#include "../../prereqs.hpp"
#include "../../core.hpp"
using namespace openml;

namespace openml{
namespace decomposition {

template<typename DataType>
class PCA {
private:
    // define matrix and vector Eigen type
    using MatType = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using VecType = Eigen::Matrix<DataType, Eigen::Dynamic, 1>;
    using IdxType = Eigen::Vector<Eigen::Index, Eigen::Dynamic>;

    /**
     * @param components: ndarray of shape (n_components, n_features)
     *      Principal axes in feature space, representing the directions of 
     *      maximum variance in the data.
     * @param explained_var: ndarray of shape (n_components,)
     *      The amount of variance explained by each of the selected components. 
     * @param explained_variance_ratio: ndarray of shape (n_components,)
     *      Percentage of variance explained by each of the selected components.
    */
    MatType components;
    VecType explained_var;
    VecType explained_var_ratio;
    std::size_t num_features;

    std::string solver;
    std::size_t num_components;
    double noise_var;

protected:

    void fit_data(const MatType& X) {
        std::size_t num_samples = X.rows(), num_features = X.cols();
        
        MatType components_;
        VecType explained_var_;
        VecType explained_var_ratio_;

        if (solver == "svd") {
            MatType U;
            VecType s; 
            MatType V;
            std::tie(U, s, V) = exact_svd<MatType, VecType>(X, false);
            
            // flip eigenvectors' sign to enforce deterministic output
            MatType Vt = V.transpose();
            std::tie(U, Vt) = math::svd_flip<MatType, VecType, IdxType>(U, Vt);

            explained_var_ = math::power<VecType>(s, 2.0) / (static_cast<DataType>(num_samples) - 1.);
            components_ = Vt;
        };

        VecType total_var = math::sum<MatType, VecType>(explained_var_);
        explained_var_ratio_ = explained_var_ / total_var(0, 0);

        if (num_components < std::min(num_samples, num_features)) {
            VecType noise_var_ = math::mean<MatType, VecType>(explained_var_.topRows(num_components));
            noise_var = static_cast<double>(noise_var_(0, 0));
        }
        else {
            noise_var = 0.0;
        }

        components = components_.topRows(num_components);
        explained_var = explained_var_.topRows(num_components);
        explained_var_ratio = explained_var_ratio_.topRows(num_components);
    }

    /** transform the new data */
    const MatType transform_data(const MatType& X) const{
        return X * components.transpose(); 
    }

    /**calculate the covariance*/
    const MatType get_data_covariance() const{
        // std::size_t num_features = components.cols();
        // MatType components_ = components;
        
        MatType explained_var_(num_components, num_components);
        explained_var_ = math::diagmat<MatType, VecType>(explained_var);

        MatType explained_var_diff = explained_var_ - noise_var;
        explained_var_diff = explained_var_diff.cwiseMax(static_cast<DataType>(0));

        MatType cov(num_features, num_features);
        cov = components.transpose() * explained_var_diff * components; 

        MatType eye(num_features, num_features);
        eye.setIdentity();
        cov += eye * noise_var;
    }

public:
    /**
     * Default constructor to create PCA object, linear dimesionality reduction using SVD 
     * of the data to project it to a lower dimesional space, input data shoule be centered 
     * 
     * @param solver the matrix decomnposition policies, 
     *      if svd, will run full svd via math::exact_svd
     * 
     * @param n_components Number of components to keep
     * @param scale Whether or not to scale the data.
    */
    PCA(): solver("svd"), 
        n_components(2) {};


    PCA(std::string solver_, 
        std::size_t num_components_): 
            solver(solver_), 
            num_components(num_components_) {};

    /**deconstructor*/
    ~PCA() {};

    /**
     * Fit the model with X.
     * @param X array-like of shape (num_samples, num_features)
     *      Training data, where num_samples is the number of 
     *      samples and num_features is the number of features.
    */
    voif fit(const MatType& X) {
        MatType centered_X;
        centered_X = math::center<MatType>(X);
        fit_data(centered_X);
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
        MatType centered_X;
        centered_X = math::center<MatType>(X);
        return transform_data(centered_X); 
    }

    /**
     * 
    */
    const MatType get_covariance() const{
        MatType cov = get_data_covariance();
        return cov;
    }

};

} // namespace openml
} // namespace regression

#endif /*METHODS_DECOMPOSITION_PCA_HPP*/
